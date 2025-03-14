import os

import denoisfm.feature_extractor as feature_extractor
import denoisfm.utils.geometry_util as geometry_util
import denoisfm.utils.shape_util as shape_util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from denoisfm.sign_correction import learned_sign_correction
from tqdm import tqdm


def load_cached_shapes(input_dir, k_eig):
    off_dir = f"{input_dir}/off"
    spectral_dir = f"{input_dir}/diffusion"

    # get all meshes in the folder
    off_files = sorted([f for f in os.listdir(off_dir) if f.endswith(".off")])

    shapes_list = []
    for file in tqdm(off_files, desc="Loading shapes and spectral operators"):
        # load the vertices and faces
        verts, faces = shape_util.read_shape(os.path.join(off_dir, file))
        verts = torch.tensor(verts, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.long)

        # load the spectral operators cached in spectral_dir
        _, mass, L, evals, evecs, gradX, gradY = geometry_util.get_operators(
            verts, faces, k=k_eig, cache_dir=spectral_dir
        )
        shapes_list.append(
            {
                "verts": verts,
                "faces": faces,
                "evecs": evecs,
                "mass": mass,
                "L": L,
                "evals": evals,
                "gradX": gradX,
                "gradY": gradY,
            }
        )
    return shapes_list


def random_signs(n):
    """
    Generate a random sign vector of length n
    """
    sign = torch.randint(0, 2, (n,))
    sign[sign == 0] = -1
    return sign.float()


if __name__ == "__main__":
    config = {
        "name": "sign_net_32_humans_anis",
        "data_dir": "/home/s94zalek_hpc/DenoisingFunctionalMaps/data/sign_training_humans_anis",
        "output_dir": "/home/s94zalek_hpc/DenoisingFunctionalMaps/checkpoints/sign_net",
        "sample_size": 32,
        "evecs_per_correc": [[32, 1]],
        # "evecs_per_correc": [[32, 1], [32, 2]],
        # "evecs_per_correc": [[32, 1], [32, 2], [32, 2]],
        "n_iter": 50000,
        "diffusionnet_params": {
            "in_channels": 128,  # 'out_channels' will be calculated automatically
            "input_type": "wks",
            "k_eig": 128,
            "n_block": 6,
        },
    }

    assert config["sample_size"] == sum(e[0] for e in config["evecs_per_correc"]), (
        "sample_size must be equal to sum of first elements in evecs_per_correc"
    )

    # calculate the number of output channels
    expected_out_channels = sum(
        evecs[0] // evecs[1] for evecs in config["evecs_per_correc"]
    )
    config["diffusionnet_params"]["out_channels"] = expected_out_channels
    

    # save the config
    exp_dir = f"{config['output_dir']}/{config['name']}"
    os.makedirs(exp_dir, exist_ok=True)

    with open(f"{exp_dir}/config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    ###################################################

    # load the shapes and spectral operators
    train_shapes = load_cached_shapes(
        input_dir=config["data_dir"],
        k_eig=config["diffusionnet_params"]["k_eig"],
    )

    # create the sign net
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sign_net = feature_extractor.DiffusionNet(**config["diffusionnet_params"]).to(
        device
    )

    # optimizer, scheduler and loss function
    opt = torch.optim.Adam(sign_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1, end_factor=0.1, total_iters=config["n_iter"]
    )
    loss_fn = torch.nn.MSELoss()

    loss_list = []

    curr_iter = 0
    train_iterator = tqdm(range(config["n_iter"]))

    while curr_iter < config["n_iter"]:
        np.random.shuffle(train_shapes)

        for curr_idx in range(len(train_shapes)):
            shape = train_shapes[curr_idx]

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

            ##############################################

            # calculate the loss
            # predicted sign difference == gt sign difference
            loss = loss_fn(signs_1 * signs_2, P_diag_1 * P_diag_2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            loss_list.append(loss.item())
            
            # save the intermediate checkpoint
            if curr_iter % (len(train_iterator) // 10) == 0:
                torch.save(sign_net.state_dict(), f"{exp_dir}/{curr_iter}.pth")
            
            # print mean of last 10 losses
            train_iterator.set_description(f"loss={np.mean(loss_list[-10:]):.3f}")
            curr_iter += 1
            train_iterator.update(1)

    # save model checkpoint
    torch.save(sign_net.state_dict(), f"{exp_dir}/{curr_iter}.pth")

    # save the loss plot
    pd.Series(loss_list).rolling(10).mean().plot()
    plt.yscale("log")
    plt.savefig(f"{exp_dir}/losses_{curr_iter}.png")
    plt.close()
