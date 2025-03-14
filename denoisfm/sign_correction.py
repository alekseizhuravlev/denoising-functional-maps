import torch


def area_normalize(Sigma, A):
    """
    Normalize the feature matrix Sigma with respect to the vertex-area matrix A using PyTorch.

    Args:
        Sigma (torch.Tensor): Feature matrix of shape (v, n), where v is the number of vertices and n is the number of features.
        A (torch.Tensor): vertex-areas of shape (v).

    Returns:
        torch.Tensor: The A-normalized feature matrix.
    """
    
    # Compute the A-norm for each column of Sigma
    norms = torch.sqrt(torch.sum((Sigma * A[:, None]) * Sigma, dim=0))  # Shape: (n,)

    # Avoid division by zero
    norms = torch.where(norms < 1e-6, torch.tensor(1.0, device=Sigma.device), norms)

    # Normalize each column of Sigma
    Sigma_normalized = Sigma / norms

    return Sigma_normalized


def interleave_corrvecs(Sigma, config):
    """
    Address the basis ambiguity by using one correction vector for several adjacent eigenvectors
    
    Example for 96 eigenvectors:
    config["evecs_per_correc"] = [
        [32, 1],      # evecs  0-31: 1 evec  / corrvec
        [32, 2],      # evecs 32-63: 2 evecs / corrvec
        [32, 4]       # evecs 64-95: 4 evecs / corrvec
    ]
    output of the feature extractor: [v x 56]
    we need to repeat-interleave the features: 
     0-31 no repeat, 
    32-47 2 times, 
    48-55 4 times
    to get [v x 96]
    """
    
    Sigma_repeated = []
    
    curr_idx = 0
    for count, factor in config["evecs_per_correc"]:
        Sigma_repeated.append(
            torch.repeat_interleave(
                Sigma[:, curr_idx:curr_idx+count // factor],
                factor, dim=-1)
        )
        curr_idx += count // factor
    
    Sigma_repeated = torch.cat(Sigma_repeated, dim=-1)
    
    assert Sigma_repeated.shape[-1] == config["sample_size"], \
        f"Shape mismatch of correction vectors: {Sigma_repeated.shape[-1]} != {config['sample_size']}"
     
    return Sigma_repeated 


def area_weighted_projection(Sigma, Phi, A):
    """
    Project the feature matrix Sigma onto the eigenvectors Phi using Sigma^T A Phi.

    Args:
        Sigma (torch.Tensor): A-normalized feature matrix of shape (v, n).
        Phi (torch.Tensor): Eigenvector matrix of shape (v, n), assumed to be A-orthonormal.
        A (torch.Tensor): vertex-areas of shape (v).

    Returns:
        torch.Tensor: Projection matrix of shape (n, n).
    """
    # Compute the projection P = Sigma^T A Phi
    P = Sigma.T @ (A[:, None] * Phi)  # Shape: (n, n)

    return P


def learned_sign_correction(
        sign_corr_net,
        shape,
        Phi,
        config,
    ):
    
    device = next(sign_corr_net.parameters()).device
    
    # get the correction vector
    Sigma = sign_corr_net(
        verts=shape["verts"].unsqueeze(0).to(device),
        faces=shape["faces"].unsqueeze(0).to(device),
        
        mass=shape["mass"].unsqueeze(0).to(device),
        L=shape["L"].unsqueeze(0).to(device),
        evals=shape["evals"][
            :sign_corr_net.k_eig
            ].unsqueeze(0).to(device),
        evecs=shape["evecs"][
            :, :sign_corr_net.k_eig
            ].unsqueeze(0).to(device),
        gradX=shape["gradX"].unsqueeze(0).to(device),
        gradY=shape["gradY"].unsqueeze(0).to(device),
    )[0]
    
    # get the eigenbasis and vertex-area matrix
    Phi = Phi.to(device)
    A = shape["mass"].to(device)

    # normalize the correction vector
    Sigma = area_normalize(Sigma, A)    
    
    # address basis ambiguity:
    # use one correction vector for several adjacent evecs
    Sigma = interleave_corrvecs(Sigma, config)   
       
    # project the correction vector onto the eigenvectors
    P = area_weighted_projection(Sigma, Phi, A)   
    
    # get diagonal elements, which are used for sign correction
    P_diag = torch.diagonal(P, dim1=0, dim2=1)
 
    return P_diag, Sigma



# Phi = shape["evecs"][:, :config["model_params"]["sample_size"]].to(device)

# correc_vector = torch.nn.functional.normalize(correc_vector, p=2, dim=0)

# normalize the evecs
# evecs_norm = torch.nn.functional.normalize(
#     shape["evecs"], p=2, dim=0
#     ).to(device)
# mass_mat = torch.diag_embed(shape["mass"]).to(device)


# projection = correction_vector @ mass @ evecs (normalized)  
# projection_mat = torch.nn.functional.normalize(
#     correc_vector_repeated.transpose(0, 1) @ mass_mat,
#     p=2, dim=-1) @ evecs_norm
