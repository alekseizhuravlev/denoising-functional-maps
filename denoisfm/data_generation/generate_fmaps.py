import argparse
import random

from ..sign_correction import learned_sign_correction, area_weighted_projection
import denoisfm.feature_extractor as feature_extractor
import denoisfm.utils.preprocessing_util as preprocessing_util
import numpy as np
import torch
import trimesh
import yaml
import logging
import utils.remesh_util as remesh_util


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

    sign_net_name = args.sign_net_name

    ### config
    exp_base_folder = f"checkpoints/sign_net/{sign_net_name}"
    with open(f"{exp_base_folder}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #######################################################
    # Model setup
    #######################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sign correction network
    sign_corr_net = feature_extractor.DiffusionNet(**config["net_params"])

    sign_corr_net.load_state_dict(
        torch.load(
            f"checkpoints/sign_net/{config['net_name']}/{config['n_iter']}.pth",
            weights_only=True,
        )
    )
    sign_corr_net.to(device)

    logging.info("Model setup finished")

    ##########################################
    # Template shape
    ##########################################

    if args.dataset_name == "SMAL_iso":
        template_path = "data/template_animal"
    else:
        template_path = "data/template_human"
        
    shape_T = trimesh.load(f"{template_path}/template.off", process=False, validate=False)
    shape_T = preprocessing_util.preprocessing_pipeline(
        shape_T.vertices, shape_T.faces, num_evecs=200, compute_distmat=False
    )
    # correspondences from template shape to SURREAL / SMAL
    shape_T["corr"] = torch.tensor(
        np.loadtxt(f"{template_path}/corr.txt") - 1
    ).long()
    
    logging.info("Template shape loaded")
    
    ##########################################
    # Load shapes
    ##########################################
    
    # load the vertices
    shapes_verts = torch.load(f"{args.input_dir}/verts.pt", mmap=True)
    
    # load the faces (same for all shapes)
    shapes_faces = torch.load(f"{args.input_dir}/faces.pt", mmap=True)
    
    logging.info("Source shapes loaded")
    
    ##########################################
    # Generate data
    ##########################################
    
    C_1T_list = []
    y_T_list = []
    y_1_list = []
    
    for i in range(args.idx_start, args.idx_end):
        
        ##########################################
        # Preprocess the training shape
        ##########################################
        
        verts, faces, corr = remesh_util.augmentation_pipeline(
            verts_orig=shapes_verts[i],
            faces_orig=shapes_faces,
            config=config_aug
        )
        shape_i = preprocessing_util.preprocessing_pipeline(
            verts, faces, num_evecs=200, compute_distmat=False
        )
        shape_i["corr"] = corr
        
        logging.info(f"{i}/{args.idx_end - args.idx_start}: remeshed and preprocessed")
        
        ##########################################
        # Obtain the sign-corrected eigenbasis and conditioning
        ##########################################
        
        # original eigenbasis
        Phi_i = shape_i["evecs"][:, :config["model_params"]["sample_size"]]
        
        # get the diagonal elements of the projection matrix and the correction vector
        with torch.no_grad():
            P_diag_i, Sigma_i = learned_sign_correction(
                sign_corr_net,
                shape_i,
                Phi_i,
                config,
            )

        signs_i = torch.sign(P_diag_i)

        # correct the eigenvectors
        Phi_i_corrected = Phi_i * signs_i

        # vertex-area matrix
        A_i = shape_i["mass"]

        # conditioning y = correction_vector @ area mat @ corrected evecs 
        y_i = area_weighted_projection(Sigma_i, Phi_i_corrected, A_i)
        
        ##########################################
        # Same for the template shape
        # (we repeat this at each iteration:
        # the output of feature extractor may be slightly different for the same shape)
        ##########################################
        
        # original eigenbasis
        Phi_T = shape_T["evecs"][:, :config["model_params"]["sample_size"]]
        
        with torch.no_grad():
            P_diag_T, Sigma_T = learned_sign_correction(
                sign_corr_net,
                shape_T,
                Phi_T,
                config,
            )
            
        signs_T = torch.sign(P_diag_T)
        
        
        Phi_T_corrected = Phi_T * signs_T
        
        A_T = shape_T["mass"]
        
        y_T = area_weighted_projection(Sigma_T, Phi_T_corrected, A_T)
        
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
        
        logging.info(f"{i}/{args.idx_end - args.idx_start}: functional map and conditioning computed")
        
        
    C_1T_list = torch.stack(C_1T_list)
    y_T_list = torch.stack(y_T_list)
    y_1_list = torch.stack(y_1_list)
    
    
    # save the results
    torch.save(C_1T_list, f"{args.output_dir}/C_1T.pt")
    torch.save(y_T_list, f"{args.output_dir}/y_T.pt")
    torch.save(y_1_list, f"{args.output_dir}/y_1.pt")
    
    logging.info("Data saved")
    
        



if __name__ == "__main__":
    
    # parameters for remeshing SMPL or SMAL shapes
    config_aug = {
        "isotropic": {
            "simplify_strength_min": 0.2, # min/max % of ALL faces to keep after simplification
            "simplify_strength_max": 0.8,
        },
        "anisotropic": {
            "probability": 0.35, # probability of applying anisotropic remeshing
                
            "fraction_to_simplify_min": 0.2, # min/max % of faces to SELECT for simplification
            "fraction_to_simplify_max": 0.6,
            
            "simplify_strength_min": 0.2, # from the SELECTED faces, min/max % to keep after simplification
            "simplify_strength_max": 0.5,
        },
    }
    
    parser = argparse.ArgumentParser(
        description="Generate a dataset of functional maps and conditioning to train a DDPM"
    )

    parser.add_argument("--sign_net_name", type=str, required=True)

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the input data: verts.pt and faces.pt")
    
    parser.add_argument("--idx_start", type=int, required=True,
                        help="Index from which to start the generation. Use for parallelization.")
    parser.add_argument("--idx_end", type=int, required=True,
                        help="Index at which to end the generation. Use for parallelization.")
    
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    run(args, config_aug)
