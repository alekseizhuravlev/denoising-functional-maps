import argparse
import logging
import random

import denoisfm.utils.geometry_util as geometry_util
import denoisfm.utils.shape_util as shape_util
import denoisfm.utils.preprocessing_util as preprocessing_util
import numpy as np
import torch
import denoisfm.utils.remesh_util as remesh_util
from tqdm import tqdm
import os


def run(args, config_aug):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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

    random_idxs = np.random.choice(len(shapes_verts), args.n_shapes, replace=False)
    
    dir_off = f"{args.output_dir}/{args.dataset_name}/off"
    dir_spectral = f"{args.output_dir}/{args.dataset_name}/diffusion"
    os.makedirs(dir_off, exist_ok=True)
    os.makedirs(dir_spectral, exist_ok=True)

    for i in tqdm(range(args.n_shapes), desc="Generating data"):
        
        verts_orig = shapes_verts[random_idxs[i]]
        faces_orig = shapes_faces

        # normalize the face area
        verts_norm = preprocessing_util.normalize_face_area(
            verts_orig,
            faces_orig
            )
        # remesh the shape
        verts_rm, faces_rm, corr = remesh_util.remesh_pipeline(
            verts_orig=verts_norm,
            faces_orig=faces_orig,
            config=config_aug,
        )
        # rotation and scaling
        verts_aug = geometry_util.data_augmentation(
            verts_rm.unsqueeze(0),
            rot_x=config_aug["extrinsic"]["rot_x"],
            rot_y=config_aug["extrinsic"]["rot_y"],
            rot_z=config_aug["extrinsic"]["rot_z"],
            std=config_aug["extrinsic"]["std"],
            scale_min=config_aug["extrinsic"]["scale_min"],
            scale_max=config_aug["extrinsic"]["scale_max"],
        )[0]

        # save the mesh
        shape_util.write_off(
            f"{dir_off}/{i:04}.off", verts_aug.cpu().numpy(), faces_rm.cpu().numpy()
        )
        
        # calculate and cache the laplacian
        geometry_util.get_operators(verts_aug, faces_rm, k=128, cache_dir=dir_spectral)


if __name__ == "__main__":
    # parameters for remeshing SMPL or SMAL shapes
    config_aug = {
        "isotropic": {
            "remesh": True, # additionally remesh before simplification 
            "simplify_strength_min": 0.2,  # min/max % of ALL faces to keep after simplification
            "simplify_strength_max": 0.8,
        },
        "anisotropic": {
            "probability": 0.0,  # probability of applying anisotropic remeshing
            "remesh": True, # additionally remesh before simplification 
            "fraction_to_simplify_min": 0.2,  # min/max % of faces to SELECT for simplification
            "fraction_to_simplify_max": 0.6,
            "simplify_strength_min": 0.2,  # from the SELECTED faces, min/max % to keep after simplification
            "simplify_strength_max": 0.5,
        },
        "extrinsic": {
            "rot_x": 0,  # random rotation around x-axis
            "rot_y": 90.0,  # random rotation around y-axis
            "rot_z": 0,
            "std": 0,  # no random noise
            "scale_min": 0.9,  # random scaling
            "scale_max": 1.1,
        },  
    }
    
    parser = argparse.ArgumentParser(
        description="Generate a dataset of meshes to train a sign correction network"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input data: verts.pt and faces.pt",
    )
    parser.add_argument("--n_shapes", type=int, required=True, default=1000)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    run(args, config_aug)
    
    # python denoisfm/data_generation/generate_meshes.py --dataset_name sign_training_humans_norm_rm --input_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL --n_shapes 1000 --output_dir /home/s94zalek_hpc/DenoisingFunctionalMaps/data
    