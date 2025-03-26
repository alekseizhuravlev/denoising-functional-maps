import denoisfm.utils.fmap_util as fmap_util
import torch
from denoisfm.sign_correction import area_weighted_projection, learned_sign_correction
from tqdm import tqdm


def get_signs_and_conditioning(
    shape,
    sign_corr_net,
    config,
):
    """
    Computes the signs of eigenvectors and conditioning matrices for a given shape using a learned sign correction network.

    Args:
        shape (dict): The shape dictionary containing vertices, faces, and spectral operators.
        sign_corr_net (torch.nn.Module): The sign correction network .
        config (dict): Configuration dictionary

    Returns:
        torch.Tensor: A tensor containing the signs for each sample.
        torch.Tensor: A tensor containing the conditioning vectors (y) for each sample.

    Notes:
        The function computes the signs and conditioning vectors multiple times (as defined by `num_samples_total`)
    """

    device = next(sign_corr_net.parameters()).device

    Phi = shape["evecs"][:, : config["ddpm_params"]["sample_size"]].to(device)
    A = shape["mass"].to(device)

    signs_list = []
    y_list = []

    for _ in range(config["inference"]["num_samples_total"]):
        with torch.no_grad():
            # get the diagonal elements of the projection matrix and the correction vector
            P_diag, Sigma = learned_sign_correction(
                sign_corr_net,
                shape,
                Phi,
                config["sign_net"],
            )

        signs = torch.sign(P_diag)

        # correct the eigenvectors
        Phi_corrected = Phi * signs

        # conditioning y = correction_vector @ area mat @ corrected evecs
        y = area_weighted_projection(Sigma, Phi_corrected, A)

        signs_list.append(signs)
        y_list.append(y)

    signs_list = torch.stack(signs_list)
    y_list = torch.stack(y_list)

    return signs_list, y_list


def template_stage(
    shape_1,
    shape_T,
    ddpm,
    sign_corr_net,
    noise_scheduler,
    config,
):
    """
    Performs the template stage of the model, where the signs of eigenvectors are corrected,
    conditioning matrices are generated, the forward diffusion process is run to obtain the template-wise functional maps,
    and the functional maps are converted to point-to-point maps.

    Args:
        shape_1 (dict): The first shape dictionary containing vertices, faces, and spectral operators.
        shape_T (dict): The template shape dictionary, same format as shape_1.
        ddpm (torch.nn.Module): The DDPM model used for denoising the samples.
        sign_corr_net (torch.nn.Module): The sign correction network.
        noise_scheduler (DDPMScheduler): The noise scheduler used during the denoising process.
        config (dict): Configuration dictionary

    Returns:
        torch.Tensor: The estimated point-to-point maps (Pi_T1) between the template and the test shape.
    """

    device = next(sign_corr_net.parameters()).device

    sample_size = config["ddpm_params"]["sample_size"]

    # get the eigenbasis of the template and the test shape
    Phi_T = shape_T["evecs"][:, :sample_size].to(device)
    Phi_1 = shape_1["evecs"][:, :sample_size].to(device)

    ###############################################
    # get signs \sigma and conditioning y, multiple times
    # ! only signs rather than the eigenvectors
    ###############################################

    signs_T_list, y_T_list = get_signs_and_conditioning(
        shape_T,
        sign_corr_net,
        config,
    )

    signs_1_list, y_1_list = get_signs_and_conditioning(
        shape_1,
        sign_corr_net,
        config,
    )

    # full conditioning: combine y for the template and the test shape
    y_full = torch.cat((y_T_list.unsqueeze(1), y_1_list.unsqueeze(1)), 1).to(device)

    ###############################################
    # Forward diffusion process
    ###############################################

    # initial noise
    x_t = torch.rand(
        config["inference"]["num_samples_total"],
        1,
        sample_size,
        sample_size,
    ).to(device)

    # Sampling loop
    for t in tqdm(noise_scheduler.timesteps, desc="Denoising..."):
        # Get model pred
        with torch.no_grad():
            residual = ddpm(x_t, t, conditioning=y_full).sample

        # Update sample with step
        x_t = noise_scheduler.step(residual, t, x_t).prev_sample

    # functional map is the last sample
    C_1T_est_list = x_t

    ##########################################################
    # Convert fmaps to template-wise point-to-point maps
    ##########################################################

    Pi_T1_list = []

    for k in range(config["inference"]["num_samples_total"]):
        # correct the eigenvectors
        Phi_T_corrected = Phi_T[:, :sample_size] * signs_T_list[k]
        Phi_1_corrected = Phi_1[:, :sample_size] * signs_1_list[k]
        C_1T_est = C_1T_est_list[k][0]

        # convert the functional map to a point-to-point map
        Pi_T1_k = fmap_util.fmap2pointmap(
            C12=C_1T_est,
            evecs_x=Phi_1_corrected,
            evecs_y=Phi_T_corrected,
        ).cpu()

        Pi_T1_list.append(Pi_T1_k)

    Pi_T1_list = torch.stack(Pi_T1_list)

    return Pi_T1_list
