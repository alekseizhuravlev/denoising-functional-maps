import torch
from torch.nn import functional as F


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
        Sigma (torch.Tensor): Feature matrix of shape (v, n).
        Phi (torch.Tensor): Eigenvector matrix of shape (v, n).
        A (torch.Tensor): vertex-areas of shape (v).

    Returns:
        torch.Tensor: Projection matrix of shape (n, n).
    """

    # compute Sigma^T A
    Sigma_A = Sigma.transpose(0, 1) @ torch.diag_embed(A)
    
    # ensure the projection values are in the range [-1, 1]
    Sigma_A_norm = F.normalize(Sigma_A, p=2, dim=1)
    Phi_norm = F.normalize(Phi, p=2, dim=0)
    
    # compute Sigma^T A Phi
    P = Sigma_A_norm @ Phi_norm
        
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
      
    # address basis ambiguity:
    # use one correction vector for several adjacent evecs
    Sigma = interleave_corrvecs(Sigma, config)   
       
    # project the correction vector onto the eigenvectors
    P = area_weighted_projection(Sigma, Phi, A)   
    
    # get diagonal elements, which are used for sign correction
    P_diag = torch.diagonal(P, dim1=0, dim2=1)
 
    return P_diag, Sigma

