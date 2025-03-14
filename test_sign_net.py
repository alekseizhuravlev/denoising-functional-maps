import logging
import random

import denoisfm.feature_extractor as feature_extractor
import numpy as np
import torch
import yaml
from denoisfm.sign_correction import learned_sign_correction
from test_on_dataset import get_dataset
from train_sign_net import random_signs
from tqdm import tqdm


def mean_sign_accuracy(signs_1, signs_2, P_diag_1, P_diag_2):
    correct_predictions = (signs_1 * signs_2).int() ==\
        torch.sign(P_diag_1) * torch.sign(P_diag_2)
    sign_accuracy = correct_predictions.float().mean()

    return sign_accuracy


def run(args):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Configuration
    exp_name = args.exp_name

    ### config
    exp_base_folder = f"checkpoints/sign_net/{exp_name}"
    with open(f"{exp_base_folder}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
        logging.FileHandler(f'{exp_base_folder}/results_{args.dataset_name}.log'),  # Save logs to a file
        logging.StreamHandler(),  # Print logs to console
    ],
    )

    #######################################################
    # Model setup
    #######################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sign_net = feature_extractor.DiffusionNet(**config["diffusionnet_params"])
    sign_net.load_state_dict(
        torch.load(
            # f"{exp_base_folder}/{config['n_iter']}.pth",
            f"{exp_base_folder}/{args.checkpoint_name}",
            weights_only=True,
        )
    )
    sign_net.to(device)

    # Test dataset
    single_dataset, _ = get_dataset(args.dataset_name, args.base_dir)
    #######################################################

    sign_accuracy_list = []

    for _ in tqdm(range(args.n_epochs)):
        for curr_idx in range(len(single_dataset)):
            shape = single_dataset[curr_idx]
            
            Phi = shape["evecs"][:, : config["sample_size"]].to(device)

            ##############################################

            # create random combilations of +1 and -1
            signs_1 = random_signs(config["sample_size"]).to(device)
            signs_2 = random_signs(config["sample_size"]).to(device)

            # evecs with random signs
            Phi_1 = Phi * signs_1
            Phi_2 = Phi * signs_2

            # get the diagonal elements of the projection matrix
            P_diag_1, _ = learned_sign_correction(
                sign_net,
                shape,
                Phi_1,
                config,
            )
            P_diag_2, _ = learned_sign_correction(
                sign_net,
                shape,
                Phi_2,
                config,
            )

            # evaluation metric
            sign_accuracy = mean_sign_accuracy(signs_1, signs_2, P_diag_1, P_diag_2)
            sign_accuracy_list.append(sign_accuracy.item())

    logging.info(f"Dataset: {args.dataset_name}, sample size: {config['sample_size']}, checkpoint {args.checkpoint_name}")
    logging.info(f"Sign accuracy: {np.mean(sign_accuracy_list) * 100:.1f}%")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=1)
    
    args = parser.parse_args()

    # python /home/s94zalek_hpc/DenoisingFunctionalMaps/test_sign_net.py --exp_name sign_net_64_humans --dataset_name FAUST_r --base_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/test --n_epochs 10

    run(args)