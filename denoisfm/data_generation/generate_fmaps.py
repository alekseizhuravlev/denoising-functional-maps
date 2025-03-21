import argparse
import logging
import os
import random

import denoisfm.feature_extractor as feature_extractor
import denoisfm.utils.preprocessing_util as preprocessing_util
import denoisfm.utils.remesh_util as remesh_util
import denoisfm.utils.visualization_util as vis_util
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import yaml
from denoisfm.sign_correction import area_weighted_projection, learned_sign_correction
import time


def visualize_before_after(idx, C_1T, C_1T_before, y_T, y_1, figures_dir):
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))

    # fmaps before and after sign correction
    vis_util.plot_fmap(axs[0], C_1T, title="after sc")
    vis_util.plot_fmap(axs[1], C_1T_before, title="before sc")

    # conditioning
    vis_util.plot_fmap(axs[2], y_T, title="y_T")
    vis_util.plot_fmap(axs[3], y_1, title="y_1")

    # save the figure
    fig.savefig(f"{figures_dir}/{idx}.png")
    plt.close(fig)


def run(args, config_aug):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    #######################################################
    # Configuration
    #######################################################

    ### sign net
    exp_base_folder = f"checkpoints/sign_net/{args.sign_net_name}"
    with open(f"{exp_base_folder}/config.yaml", "r") as f:
        config_sign_net = yaml.load(f, Loader=yaml.FullLoader)

    #######################################################
    # Model setup
    #######################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sign correction network
    sign_corr_net = feature_extractor.DiffusionNet(
        **config_sign_net["diffusionnet_params"]
    )

    sign_corr_net.load_state_dict(
        torch.load(
            f"checkpoints/sign_net/{config_sign_net['name']}/{config_sign_net['n_iter']}.pth",
            weights_only=True,
            map_location=device,
        )
    )
    sign_corr_net.to(device)

    logging.info("Model setup finished")

    ##########################################
    # Template shape
    ##########################################

    template_path = f"data/template/{args.template_type}"

    shape_T = trimesh.load(
        f"{template_path}/template.off", process=False, validate=False
    )
    shape_T = preprocessing_util.preprocessing_pipeline(
        shape_T.vertices, shape_T.faces, num_evecs=200, compute_distmat=False
    )
    # correspondences from template shape to SURREAL / SMAL
    shape_T["corr"] = torch.tensor(np.loadtxt(f"{template_path}/corr.txt") - 1).long()

    logging.info("Template shape loaded")

    ##########################################
    # Load shapes
    ##########################################

    # load the vertices
    shapes_verts = torch.load(f"{args.input_dir}/verts.pt", mmap=True)

    # load the faces (same for all shapes)
    shapes_faces = torch.load(f"{args.input_dir}/faces.pt")

    logging.info("Source shapes loaded")

    ##########################################
    # Prepare the directories
    ##########################################

    save_dir = f"{args.output_dir}/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)

    config_dataset = {
        "template_type": args.template_type,
        "augmentations": config_aug,
        "sign_net": config_sign_net,
    }

    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(config_dataset, f, sort_keys=False)

    ##########################################
    # Generate data
    ##########################################
    
    time_start = time.time()

    C_1T_list = []
    y_T_list = []
    y_1_list = []

    for i in range(args.idx_start, args.idx_end):
        ##########################################
        # Preprocess the training shape
        ##########################################

        verts, faces, corr = remesh_util.remesh_pipeline(
            verts_orig=shapes_verts[i], faces_orig=shapes_faces, config=config_aug
        )
        shape_i = preprocessing_util.preprocessing_pipeline(
            verts, faces, num_evecs=200, compute_distmat=False
        )
        shape_i["corr"] = corr

        ##########################################
        # Obtain the sign-corrected eigenbasis and conditioning
        ##########################################

        # original eigenbasis
        Phi_i = shape_i["evecs"][:, : config_sign_net["sample_size"]]
        Phi_i = Phi_i.to(device)

        # get the diagonal elements of the projection matrix and the correction vector
        with torch.no_grad():
            P_diag_i, Sigma_i = learned_sign_correction(
                sign_corr_net,
                shape_i,
                Phi_i,
                config_sign_net,
            )

        signs_i = torch.sign(P_diag_i)

        # correct the eigenvectors
        Phi_i_corrected = Phi_i * signs_i

        # vertex-area matrix
        A_i = shape_i["mass"].to(device)

        # conditioning y = correction_vector @ area mat @ corrected evecs
        y_i = area_weighted_projection(Sigma_i, Phi_i_corrected, A_i).cpu()

        ##########################################
        # Same for the template shape
        # we repeat this at each iteration:
        # the output of feature extractor may be slightly different for the same shape
        ##########################################

        # original eigenbasis
        Phi_T = shape_T["evecs"][:, : config_sign_net["sample_size"]]
        Phi_T = Phi_T.to(device)

        with torch.no_grad():
            P_diag_T, Sigma_T = learned_sign_correction(
                sign_corr_net,
                shape_T,
                Phi_T,
                config_sign_net,
            )

        signs_T = torch.sign(P_diag_T)

        Phi_T_corrected = Phi_T * signs_T

        A_T = shape_T["mass"].to(device)

        y_T = area_weighted_projection(Sigma_T, Phi_T_corrected, A_T).cpu()

        ##########################################
        # Functional map after sign correction
        ##########################################

        C_1T = torch.linalg.lstsq(
            Phi_T_corrected[shape_T["corr"]].to(device),
            Phi_i_corrected[shape_i["corr"]].to(device),
        ).solution.cpu()

        C_1T_list.append(C_1T)
        y_T_list.append(y_T)
        y_1_list.append(y_i)
        
        curr_iter = i - args.idx_start
        
        # log the progress every 50 iterations
        if (curr_iter) % 50 == 0:
            time_elapsed = time.time() - time_start
            logging.info(
                f"{curr_iter}/{args.idx_end - args.idx_start}: C_1T and y computed, avg time: {time_elapsed / (curr_iter + 1):.1f}s"
            )

        # visualize the data
        if curr_iter % args.vis_freq == 0:
            C_1T_before = torch.linalg.lstsq(
                Phi_T[shape_T["corr"]].to(device),
                Phi_i[shape_i["corr"]].to(device),
            ).solution.cpu()

            visualize_before_after(
                i, C_1T, C_1T_before, y_T, y_i, f"{save_dir}/figures"
            )

    C_1T_list = torch.stack(C_1T_list)
    y_T_list = torch.stack(y_T_list)
    y_1_list = torch.stack(y_1_list)

    # save the results with start and end indices
    torch.save(C_1T_list, f"{save_dir}/C_1T_{args.idx_start}_{args.idx_end}.pt")
    torch.save(y_T_list, f"{save_dir}/y_T_{args.idx_start}_{args.idx_end}.pt")
    torch.save(y_1_list, f"{save_dir}/y_1_{args.idx_start}_{args.idx_end}.pt")

    logging.info("Data saved")


if __name__ == "__main__":
    # parameters for remeshing SMPL or SMAL shapes
    config_aug = {
        "isotropic": {
            "remesh": True,
            "simplify_strength_min": 0.2,  # min/max % of ALL faces to keep after simplification
            "simplify_strength_max": 0.8,
        },
        "anisotropic": {
            "probability": 0.35,  # probability of applying anisotropic remeshing
            "remesh": True,
            "fraction_to_simplify_min": 0.2,  # min/max % of faces to SELECT for simplification
            "fraction_to_simplify_max": 0.6,
            "simplify_strength_min": 0.2,  # from the SELECTED faces, min/max % to keep after simplification
            "simplify_strength_max": 0.5,
        },
    }

    parser = argparse.ArgumentParser(
        description="Generate a dataset of functional maps and conditioning to train a DDPM"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset that will be generated",
    )

    parser.add_argument("--sign_net_name", type=str, required=True)

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input data: verts.pt and faces.pt",
    )
    parser.add_argument(
        "--template_type", type=str, required=True, choices=["human", "animal"]
    )

    parser.add_argument(
        "--idx_start",
        type=int,
        required=True,
        help="Index from which to start the generation. Use for parallelization.",
    )
    parser.add_argument(
        "--idx_end",
        type=int,
        required=True,
        help="Index at which to end the generation. Use for parallelization.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the generated data will be saved with the dataset_name",
    )
    parser.add_argument(
        "--vis_freq",
        type=int,
        default=1000,
        help="Frequency at which to visualize fmaps and conditioning",
    )

    args = parser.parse_args()

    run(args, config_aug)

    # python /home/s94zalek_hpc/DenoisingFunctionalMaps/denoisfm/data_generation/generate_fmaps.py --sign_net_name sign_net_64_norm_rm --input_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL --idx_start 0 --idx_end 100 --output_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL_sign_net_64_norm_rm --vis_freq 1 --template_type human
