
import denoisfm.map_selection as map_selection
import denoisfm.utils.fmap_util as fmap_util
import torch


def template_map_to_pairwise_map(
    Pi_T1, Pi_T2,
    Phi_1, Phi_2,
    A_2,
    sample_size,
    zoomout_step
    ):
    """
    Converts template-wise point-to-point maps into a pairwise map, refines them 
    with the ZoomOut technique and then converts into a point-to-point map.

    Args:
        Pi_T1 (torch.Tensor): A tensor representing the template-wise point-to-point map from shape 1 to shape T.
        Pi_T2 (torch.Tensor): A tensor representing the template-wise point-to-point map from shape 2 to shape T.
        Phi_1 (torch.Tensor): The eigenbasis of shape 1.
        Phi_2 (torch.Tensor): The eigenbasis of shape 2.
        A_2 (torch.Tensor): The vertex-area matrix of shape 2.
        sample_size (int): The number of eigenvectors used in the functional map calculation.
        zoomout_step (int): The step size for the ZoomOut technique to refine the functional map.

    Returns:
        torch.Tensor: The refined pairwise point-to-point map between shape 1 and shape 2.
    """
    
    # assert that the number of eigenvectors - sample_size is divisible by zoomout_step
    assert (Phi_1.shape[1] - sample_size) % zoomout_step == 0, \
        f'Number of eigenvectors to zoom out must be divisible by zoomout_step: {Phi_1.shape[1] - sample_size} % {zoomout_step} != 0'
    
    # get the functional map from shape 1 to shape 2
    # with the original sample size, e.g. 64x64
    C_12 = torch.linalg.lstsq(
        Phi_2[:, :sample_size][Pi_T2],
        Phi_1[:, :sample_size][Pi_T1]
        ).solution
    
    # upsample the functional map with ZoomOut to 200x200
    C_12_zo = fmap_util.zoomout(
        FM_12=C_12, 
        evects1=Phi_1,
        evects2=Phi_2,
        nit=(Phi_1.shape[1] - sample_size) // zoomout_step,
        step=zoomout_step,
        A2=A_2
    )

    # convert to a point-to-point map
    Pi_21 = fmap_util.fmap2pointmap(
        C12=C_12_zo,
        evecs_x=Phi_1,
        evecs_y=Phi_2,
        ).cpu()
    
    return Pi_21



def pairwise_stage(
    shape_1, shape_2,
    Pi_T1_list, Pi_T2_list,
    config
):
    """
    Convert the template-wise maps to pairwise, refines them using ZoomOut technique 
    and selects a medoid map from the generated pairwise maps.

    Args:
        shape_1 (dict): A dictionary representing shape 1, containing the vertices, faces, and spectral operators.
        shape_2 (dict): A dictionary representing shape 2, same format as shape_1.
        Pi_T1_list (torch.Tensor): A tensor containing a list of template-wise point-to-point maps for shape 1.
        Pi_T2_list (torch.Tensor): A tensor containing a list of template-wise point-to-point maps for shape 2.
        config (dict): A dictionary containing configuration parameters
        
    Returns:
        torch.Tensor: The medoid point-to-point map between shape 1 and shape 2.
    """
           
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # full eigenbasis (200) of shapes 1 and 2 
    Phi_1 = shape_1['evecs'].to(device)
    Phi_2 = shape_2['evecs'].to(device)
    
    # vertex-area matrix of shape 2
    A_2 = shape_2['mass'].to(device)
    
    # template-wise point-to-point maps
    Pi_T1_list = Pi_T1_list.to(device)
    Pi_T2_list = Pi_T2_list.to(device)
    
    ###############################################
    # Convert template-wise maps to pairwise ones
    # and refine with ZoomOut
    ###############################################
    
    Pi_21_list = []
    for k in range(Pi_T1_list.shape[0]):
        
        Pi_21_k = template_map_to_pairwise_map(
            Pi_T1_list[k], Pi_T2_list[k],
            Phi_1, Phi_2, A_2,
            sample_size=config["ddpm_params"]["sample_size"],
            zoomout_step=config["inference"]["zoomout_step"],
            )
        Pi_21_list.append(Pi_21_k)
        
    Pi_21_list = torch.stack(Pi_21_list)
       
    ###############################################
    # medoid map selection
    ###############################################

    Pi_21_medoid = map_selection.select_p2p_map(
        Pi_21_list.to(device),
        shape_1['verts'].to(device),
        shape_2['L'].to(device), 
        shape_1['dist'].to(device),
        num_samples_selection=config["inference"]["num_samples_selection"]
        ).cpu()

    return Pi_21_medoid
