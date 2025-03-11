import argparse
import os
import random

import accelerate
import DenoisingFunctionalMaps.denoisfm.feature_extractor as feature_extractor
import denoisfm.utils.fmap_util as fmap_util
import denoisfm.utils.preprocessing_util as preprocessing_util
import numpy as np
import torch
import trimesh
import yaml
from DenoisingFunctionalMaps.denoisfm.conditional_unet import ConditionalUnet
from DenoisingFunctionalMaps.denoisfm.sign_correction import predict_sign_change

# models
from diffusers import DDPMScheduler
from tqdm import tqdm



def run(args):
    # set random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    #######################################################
    # Configuration
    #######################################################

    experiment_name = args.experiment_name
    checkpoint_name = args.checkpoint_name

    ### config
    exp_base_folder = f"checkpoints/ddpm/{experiment_name}"
    with open(f"{exp_base_folder}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #######################################################
    # Model setup
    #######################################################

    ### DDPM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalUnet(config["model_params"])

    if "accelerate" in config and config["accelerate"]:
        accelerate.load_checkpoint_in_model(
            model, f"{exp_base_folder}/checkpoints/{checkpoint_name}/model.safetensors"
        )
    else:
        model.load_state_dict(
            torch.load(
                f"{exp_base_folder}/checkpoints/{checkpoint_name}", weights_only=True
            )
        )

    model.to(device)

    ### Sign correction network
    sign_corr_net = feature_extractor.DiffusionNet(**config["sign_net"]["net_params"])
    sign_corr_net.load_state_dict(
        torch.load(
            f"checkpoints/sign_net/{config['sign_net']['net_name']}/{config['sign_net']['n_iter']}.pth",
            weights_only=True,
        )
    )
    sign_corr_net.to(device)

    ### noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True
    )

    ##########################################
    # Template shape
    ##########################################

    template_shape = trimesh.load("data/template/template.off", process=False)
    template_shape = preprocessing_util.preprocessing_pipeline(
        template_shape.vertices, template_shape.faces, num_evecs=128, centering="bbox"
    )

    ##########################################
    # Test shape
    ##########################################

    test_shape = trimesh.load(args.test_shape, process=False)
    test_shape = preprocessing_util.preprocessing_pipeline(
        test_shape.vertices, test_shape.faces, num_evecs=128, centering="bbox"
    )

    ##########################################
    # Template stage
    ##########################################

    p2p_maps_template = template_stage(
        model,
        sign_corr_net,
        noise_scheduler,
        test_shape,
        template_shape,
        config,
        num_iters_avg=args.num_iters_avg,
    )

    ##########################################
    # Save the template-wise maps
    ##########################################

    # get name of test shape
    test_shape_name = os.path.basename(args.test_shape).replace(".off", "")
    save_path = f"results/template_stage_{test_shape_name}.pt"

    torch.save(p2p_maps_template, save_path)


def template_stage():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model")

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)

    parser.add_argument("--test_shape", type=str, required=True)

    parser.add_argument("--num_iters_avg", type=int, default=128)

    args = parser.parse_args()

    run(args)
