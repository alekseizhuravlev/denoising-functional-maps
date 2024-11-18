import torch

def predict_sign_change(net, verts, faces, evecs_input, mass_mat,
                        input_type, evecs_per_correc, input_feats=None,
                        **kwargs):
    
    # check the input
    assert verts.dim() == 3
    assert faces is None or faces.dim() == 3
    assert evecs_input.dim() == 3
    assert mass_mat is None or (mass_mat.dim() == 3 and mass_mat.shape[1] == mass_mat.shape[2])
    
    # normalize the evecs
    evecs_input = torch.nn.functional.normalize(evecs_input, p=2, dim=1)
    
    
    ##################################################
    # get input for the network
    ##################################################
    
    if input_type == 'wks' or input_type == 'xyz':
        # will be computed automatically
        input_feats = None
        
    elif input_type == 'evecs':
        input_feats = evecs_input
        
    else:
        raise ValueError(f'Unknown input type {input_type}')
        
    
    ##################################################
    # get the correc vector
    ##################################################
    
    correc_vector_flip = net(
        verts=verts,
        faces=faces,
        feats=input_feats,
        **kwargs
    ) # [1 x 6890 x 1]

    # normalize the correc vector
    correc_vector_norm = torch.nn.functional.normalize(correc_vector_flip, p=2, dim=1)
    
    
    ##################################################
    # address basis ambiguity: 
    # use one correc vector for several adjacent evecs
    ##################################################
    
    if correc_vector_norm.shape[-1] == evecs_input.shape[-1]:
        
        # one correc vector for each evec
        correc_vector_norm_repeated = correc_vector_norm
    
    elif isinstance(evecs_per_correc, int):
              
        # x evecs per correc vector
        
        assert evecs_input.shape[-1] % correc_vector_norm.shape[-1] == 0
        
        repeat_factor = evecs_input.shape[-1] // correc_vector_norm.shape[-1]
        correc_vector_norm_repeated = torch.repeat_interleave(
            correc_vector_norm, repeat_factor, dim=-1)
        
    elif isinstance(evecs_per_correc, tuple):
               
        # 0-31: x evecs for correc vector
        # 32-63: y evecs for correc vector
        # etc.
        
        assert evecs_input.shape[-1] % len(evecs_per_correc) == 0
        
        segment_size = evecs_input.shape[-1] // len(evecs_per_correc)
        # 64 // 2 = 32
        
        correc_vector_segments = []
        for evecs_num in evecs_per_correc:
            assert segment_size % evecs_num == 0
            correc_vector_segments.append(segment_size // evecs_num)
            # 32 // 1 = 32, 32 // 2 = 16 -> [32, 16]
            
        # repeat the correc vector x times for each entry in evecs_per_correc
        correc_vector_norm_repeated = torch.tensor([], device=correc_vector_norm.device)
        current_idx = 0
        
        for i in range(len(evecs_per_correc)):

            correc_vector_repeated_i = torch.repeat_interleave(
                correc_vector_norm[:, :, current_idx:current_idx+correc_vector_segments[i]],
                evecs_per_correc[i], dim=-1)
            
            correc_vector_norm_repeated = torch.cat([
                correc_vector_norm_repeated,
                correc_vector_repeated_i], dim=-1)
            
            current_idx += correc_vector_segments[i]
        
    else:
        raise ValueError(f'Unknown evecs_per_correc type {evecs_per_correc}')
        
    
    # mass-normalize the correc vector and project it onto evecs    
    correc_mass_norm = torch.nn.functional.normalize(
        correc_vector_norm_repeated.transpose(1, 2) @ mass_mat,
        p=2, dim=2)
    
    product_with_correc = correc_mass_norm @ evecs_input
        

    
    assert product_with_correc.shape[1] == product_with_correc.shape[2]
    
    # get of diagonal elements
    projection_values = torch.diagonal(product_with_correc, dim1=1, dim2=2)
 
    return projection_values, correc_vector_norm_repeated

