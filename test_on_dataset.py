import argparse
import logging
import os
import random

import accelerate
import denoisfm.conditional_unet as conditional_unet
import denoisfm.feature_extractor as feature_extractor
import denoisfm.shape_dataset as shape_dataset
import denoisfm.utils.preprocessing_util as preprocessing_util
import numpy as np
import torch
import trimesh
import yaml
from denoisfm.inference_pairwise_stage import pairwise_stage
from denoisfm.inference_template_stage import template_stage
from denoisfm.utils.geodist_util import calculate_geodesic_error
from diffusers import DDPMScheduler
from tqdm import tqdm


def get_dataset(name, base_dir):
    """
    Retrieves a testing dataset based on the specified name and base directory.

    Args:
        name (str): Must be one of the following: "FAUST_r", "SCAPE_r", "SHREC19_r", "FAUST_a", "SCAPE_a", "DT4D_intra", "DT4D_inter", "SMAL_iso".
        base_dir (str): The base directory where dataset files are stored.

    Returns:
        tuple: A tuple containing:
            - single_dataset: A dataset containing single shapes.
            - pair_dataset: A dataset containing pairs of shapes.
    """
    # dictionary mapping dataset names to their corresponding classes and parameters
    dataset_configs = {
        "FAUST_r": (
            shape_dataset.PairFaustDataset,
            {"data_root": f"{base_dir}/FAUST_r", "phase": "test"},
        ),
        "SCAPE_r": (
            shape_dataset.PairScapeDataset,
            {"data_root": f"{base_dir}/SCAPE_r", "phase": "test"},
        ),
        "SHREC19_r": (
            shape_dataset.PairShrec19Dataset,
            {"data_root": f"{base_dir}/SHREC19_r", "phase": "test"},
        ),
        "FAUST_a": (shape_dataset.PairDataset, {"data_root": f"{base_dir}/FAUST_a"}),
        "SCAPE_a": (shape_dataset.PairDataset, {"data_root": f"{base_dir}/SCAPE_a"}),
        "DT4D_intra": (
            shape_dataset.PairDT4DDataset,
            {"data_root": f"{base_dir}/DT4D_r", "phase": "test", "inter_class": False},
        ),
        "DT4D_inter": (
            shape_dataset.PairDT4DDataset,
            {"data_root": f"{base_dir}/DT4D_r", "phase": "test", "inter_class": True},
        ),
        "SMAL_iso": (
            shape_dataset.PairSmalDataset,
            {"data_root": f"{base_dir}/SMAL_r", "phase": "test", "category": False},
        ),
    }

    if name not in dataset_configs:
        raise ValueError(f"Dataset {name} not found")

    # Retrieve and instantiate the dataset class
    dataset_class, kwargs = dataset_configs[name]
    pair_dataset = dataset_class(**kwargs)

    # Extract single dataset from pair_dataset
    single_dataset = pair_dataset.dataset

    return single_dataset, pair_dataset


def run(args):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    #######################################################
    # Configuration
    #######################################################

    exp_name = args.exp_name

    ### config
    exp_base_dir = f"checkpoints/ddpm/{exp_name}"
    with open(f"{exp_base_dir}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    output_dir = f"results/{exp_name}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/p2p", exist_ok=True)


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                f"{output_dir}/results_{args.dataset_name}.log"
            ),  # Save logs to a file
            logging.StreamHandler(),  # Print logs to console
        ],
    )

    #######################################################
    # Model setup
    #######################################################

    # DDPM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = conditional_unet.ConditionalUnet(config["ddpm_params"])
    checkpoint_name = args.checkpoint_name

    accelerate.load_checkpoint_in_model(
        ddpm, f"{exp_base_dir}/checkpoints/{checkpoint_name}/model.safetensors"
    )
    ddpm.to(device)

    # noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True
    )

    # sign correction network
    sign_corr_net = feature_extractor.DiffusionNet(
        **config["sign_net"]["diffusionnet_params"]
    )

    sign_corr_net.load_state_dict(
        torch.load(
            f"checkpoints/sign_net/{config['sign_net']['name']}/{config['sign_net']['n_iter']}.pth",
            weights_only=True,
        )
    )
    sign_corr_net.to(device)

    logging.info("Model setup finished")

    ##########################################
    # Template shape
    ##########################################

    shape_T = trimesh.load(
        f"data/template/{config['template_type']}/template.off",
        process=False,
        validate=False,
    )
    shape_T = preprocessing_util.preprocessing_pipeline(
        shape_T.vertices, shape_T.faces, num_evecs=200, compute_distmat=False
    )
    logging.info(f"Template shape loaded, type: {config['template_type']}")

    ##########################################
    # Test dataset
    ##########################################

    single_dataset, pair_dataset = get_dataset(args.dataset_name, args.data_dir)
    logging.info(f"Dataset {args.dataset_name} loaded")

    ##########################################
    # Template stage
    ##########################################

    Pi_Ti_list = []

    for i in tqdm(range(len(single_dataset)), desc="Template stage"):
        shape_i = single_dataset[i]

        # get template-wise maps for the shape
        Pi_Ti = template_stage(
            shape_i,
            shape_T,
            ddpm,
            sign_corr_net,
            noise_scheduler,
            config,
        )
        Pi_Ti_list.append(Pi_Ti)

    logging.info("Template stage finished")

    ##########################################
    # Pairwise stage
    ##########################################

    geo_err_list = []

    for i in tqdm(range(len(pair_dataset)), desc="Pairwise stage"):
        pair_i = pair_dataset[i]
        shape_1, shape_2 = pair_i["first"], pair_i["second"]

        # get the template-wise maps for the two shapes
        Pi_T1 = Pi_Ti_list[shape_1["id"]]
        Pi_T2 = Pi_Ti_list[shape_2["id"]]

        # convert the template-wise maps to pairwise ones, apply post-processing
        Pi_21 = pairwise_stage(shape_1, shape_2, Pi_T1, Pi_T2, config)

        # save the pairwise map
        name_1, name_2 = shape_1["name"], shape_2["name"]
        output_file = f"{output_dir}/p2p/{name_1}_{name_2}.pt"
        torch.save(Pi_21, output_file)

        # calculate the geodesic error
        geo_err = (
            calculate_geodesic_error(
                shape_1["dist"],
                shape_1["corr"],
                shape_2["corr"],
                Pi_21,
                return_mean=True,
            )
            * 100
        )
        geo_err_list.append(geo_err)

        logging.info(f"Geodesic error for {name_1} and {name_2}: {geo_err:.1f}")

    logging.info("Pairwise stage finished")

    geo_err_list = torch.tensor(geo_err_list)
    logging.info(
        f"Mean geodesic error on {args.dataset_name}: {torch.mean(geo_err_list):.1f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on a dataset")
    dataset_choices = [
        "FAUST_r",
        "SCAPE_r",
        "SHREC19_r",
        "FAUST_a",
        "SCAPE_a",
        "DT4D_intra",
        "DT4D_inter",
        "SMAL_iso",
    ]

    parser.add_argument("--exp_name", type=str, required=True,
                        help="Name of the experiment, e.g. ddpm_64"
                        )
    parser.add_argument("--checkpoint_name", type=str, required=True,
                        help="Name of the checkpoint, e.g. epoch_99")
    parser.add_argument(
        "--dataset_name", type=str, choices=dataset_choices, required=True,
        help="Name of the test dataset, e.g. FAUST_r"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset files, e.g. data/test",
    )

    args = parser.parse_args()

    run(args)
