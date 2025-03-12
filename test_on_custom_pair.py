import argparse
import os
import random

import accelerate
import denoisfm.conditional_unet as conditional_unet
import denoisfm.feature_extractor as feature_extractor
import denoisfm.utils.preprocessing_util as preprocessing_util
import numpy as np
import torch
import trimesh
import yaml
from denoisfm.inference_pairwise_stage import pairwise_stage
from denoisfm.inference_template_stage import template_stage
from diffusers import DDPMScheduler
import logging



def run(args):
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    #######################################################
    # Configuration
    #######################################################

    exp_name = args.exp_name

    ### config
    exp_base_folder = f"checkpoints/ddpm/{exp_name}"
    with open(f"{exp_base_folder}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #######################################################
    # Model setup
    #######################################################

    # DDPM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = conditional_unet.ConditionalUnet(config["model_params"])
    checkpoint_name = config["checkpoint_name"]

    if "accelerate" in config and config["accelerate"]:
        accelerate.load_checkpoint_in_model(
            ddpm,
            f"{exp_base_folder}/checkpoints/{checkpoint_name}/model.safetensors"
        )
    else:
        ddpm.load_state_dict(torch.load(
            f"{exp_base_folder}/checkpoints/{checkpoint_name}", 
            weights_only=True))

    ddpm.to(device)
    
    # noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True
    )

    # sign correction network
    sign_corr_net = feature_extractor.DiffusionNet(**config["sign_net"]["net_params"])
    
    sign_corr_net.load_state_dict(torch.load(
        f"checkpoints/sign_net/{config['sign_net']['net_name']}/{config['sign_net']['n_iter']}.pth",
        weights_only=True))
    sign_corr_net.to(device)

    logging.info("Model setup finished")

    ##########################################
    # Template shape
    ##########################################

    shape_T = trimesh.load(
        "data/template/template.off", process=False, validate=False
    )
    shape_T = preprocessing_util.preprocessing_pipeline(
        shape_T.vertices, shape_T.faces, num_evecs=200, compute_distmat=False
    )
    logging.info("Template shape loaded")

    ##########################################
    # Test shapes
    ##########################################

    shape_1 = trimesh.load(args.shape_1, process=False, validate=False)
    shape_2 = trimesh.load(args.shape_2, process=False, validate=False)

    shape_1 = preprocessing_util.preprocessing_pipeline(
        shape_1.vertices, shape_1.faces, num_evecs=200, compute_distmat=True
    )
    shape_2 = preprocessing_util.preprocessing_pipeline(
        shape_2.vertices, shape_2.faces, num_evecs=200, compute_distmat=False
    ) 
    logging.info("Shapes 1 and 2 loaded")

    ##########################################
    # Template stage
    ##########################################

    Pi_T1 = template_stage(
        shape_1,
        shape_T,
        ddpm,
        sign_corr_net,
        noise_scheduler,
        config,
    )
    logging.info("Template stage for shape 1 finished")
    
    Pi_T2 = template_stage(
        shape_2,
        shape_T,
        ddpm,
        sign_corr_net,
        noise_scheduler,
        config,
    )
    logging.info("Template stage for shape 2 finished")
    
    # save the template-wise maps
    name_1 = os.path.splitext(os.path.basename(args.shape_1))[0]
    name_2 = os.path.splitext(os.path.basename(args.shape_2))[0]
    
    torch.save(Pi_T1, f"results/{name_1}_template.pt")
    torch.save(Pi_T2, f"results/{name_2}_template.pt")

    ##########################################
    # Pairwise stage
    ##########################################

    Pi_21 = pairwise_stage(
        shape_1, shape_2, Pi_T1, Pi_T2, config
    )
    logging.info("Pairwise stage finished")

    # get the name of the test shapes, remove the extension

    output_file = f"results/{name_1}_{name_2}.pt"

    torch.save(Pi_21, output_file)
    logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the model on a custom pair of shapes"
    )

    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--shape_1", type=str, required=True)
    parser.add_argument("--shape_2", type=str, required=True)

    args = parser.parse_args()

    run(args)
