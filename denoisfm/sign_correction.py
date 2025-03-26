import torch
import torch.nn.functional as F


def interleave_corrvecs(Sigma, config):
    """
    Address the basis ambiguity by using one correction vector for several adjacent eigenvectors.

    This function interleaves the correction vectors according to the configuration provided.
    It repeats specific segments of the correction vector a given number of times to align with the
    corresponding eigenvectors.

    Example for 96 eigenvectors:
    config["evecs_per_correc"] = [
        [32, 1],      # evecs  0-31: 1 evec  / corrvec
        [32, 2],      # evecs 32-63: 2 evecs / corrvec
        [32, 4]       # evecs 64-95: 4 evecs / corrvec
    ]

    Output of the feature extractor: (V, 56)
    We need to repeat-interleave the features:
     - 0-31: no repeat,
     - 32-47: 2 times,
     - 48-55: 4 times

    to get (V, 96).

    Args:
        Sigma (torch.Tensor): Correction vector of shape (V, M), where V is the number of vertices and M is the reduced feature dimension.
        config (dict): Configuration dictionary with the following keys:
            - "evecs_per_correc": List of [count, factor] specifying how many eigenvectors to map to each correction vector.
            - "sample_size": Total number of eigenvectors (N) after interleaving.

    Returns:
        torch.Tensor: Interleaved correction vector of shape (V, N).
    """

    Sigma_repeated = []

    curr_idx = 0
    # Iterate through each segment defined in "evecs_per_correc"
    for count, factor in config["evecs_per_correc"]:
        # Select the relevant portion of Sigma and repeat-interleave it by the specified factor
        Sigma_repeated.append(
            torch.repeat_interleave(
                Sigma[:, curr_idx : curr_idx + count // factor], factor, dim=-1
            )
        )
        # Move the starting index forward by the number of features added
        curr_idx += count // factor

    # Concatenate all repeated segments along the feature dimension
    Sigma_repeated = torch.cat(Sigma_repeated, dim=-1)

    # Ensure the resulting tensor matches the expected sample size
    assert Sigma_repeated.shape[-1] == config["sample_size"], (
        f"Shape mismatch of correction vectors: {Sigma_repeated.shape[-1]} != {config['sample_size']}"
    )

    return Sigma_repeated


def area_weighted_projection(Sigma, Phi, A):
    """
    Project the feature matrix Sigma onto the eigenvectors Phi using Sigma^T A Phi.

    Args:
        Sigma (torch.Tensor): A-normalized feature matrix of shape (V, N).
        Phi (torch.Tensor): Eigenvector matrix of shape (V, N), assumed to be A-orthonormal.
        A (torch.Tensor): vertex-areas of shape (V).

    Returns:
        torch.Tensor: Projection matrix of shape (N, N).
    """

    A_mat = torch.diag_embed(A)

    P = F.normalize(Sigma.transpose(0, 1) @ A_mat, p=2, dim=1) @\
        F.normalize(Phi, p=2, dim=0)

    return P


def learned_sign_correction(
    sign_corr_net,
    shape,
    Phi,
    config,
):
    """
    Apply learned sign correction to the eigenvectors using a neural network.

    Args:
        sign_corr_net (torch.nn.Module): Neural network model to predict the correction vector.
        shape (dict): Dictionary containing shape information with the following keys: "verts", "faces", "mass", "L", "evals", "evecs", "gradX", "gradY".
        Phi (torch.Tensor): Eigenvector matrix of shape (V, N).
        config (dict): Configuration dictionary of the sign correction network with key "evecs_per_correc" for interleaving.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - P_diag: Diagonal elements of the projection matrix (N,).
            - Sigma: The learned and interleaved correction vector (V, N).
    """

    device = next(sign_corr_net.parameters()).device

    # get the correction vector
    Sigma = sign_corr_net(
        verts=shape["verts"].unsqueeze(0).to(device),
        faces=shape["faces"].unsqueeze(0).to(device),
        mass=shape["mass"].unsqueeze(0).to(device),
        L=shape["L"].unsqueeze(0).to(device),
        evals=shape["evals"][: sign_corr_net.k_eig].unsqueeze(0).to(device),
        evecs=shape["evecs"][:, : sign_corr_net.k_eig].unsqueeze(0).to(device),
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
