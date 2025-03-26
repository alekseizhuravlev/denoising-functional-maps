import argparse
import os

import torch
import yaml
from accelerate import Accelerator
from denoisfm.conditional_unet import ConditionalUnet
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
import random
import numpy as np


class DatasetDDPM(torch.utils.data.Dataset):
    def __init__(self, base_dir):
        super().__init__()

        self.C_1T = torch.load(f"{base_dir}/C_1T.pt", mmap=True, weights_only=True)
        self.y_T = torch.load(f"{base_dir}/y_T.pt", mmap=True, weights_only=True)
        self.y_1 = torch.load(f"{base_dir}/y_1.pt", mmap=True, weights_only=True)

        assert self.C_1T.dtype == self.y_T.dtype == self.y_1.dtype == torch.float32

    def __len__(self):
        return len(self.C_1T)

    def __getitem__(self, idx):
        C_1T = self.C_1T[idx].unsqueeze(0)
        y_full = torch.stack((self.y_T[idx], self.y_1[idx])).contiguous()

        return C_1T, y_full


def run(args):
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # configuration
    config = {
        "exp_name": args.exp_name,
        "dataset_base_dir": "/tmp",
        "dataset_name": args.dataset_name,
        "n_epochs": 100,
        "checkpoint_every": 10,
        "batch_size": 64,
        "ddpm_params": {
            "sample_size": args.sample_size,
            "in_channels": 3,
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": tuple(map(int, args.block_out_channels.split(","))),
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
        },
        "inference": {
            "num_samples_total": 128,
            "num_samples_selection": 16,
            "zoomout_step": 1,
        },
    }

    # experiment setup
    exp_dir = f"checkpoints/ddpm/{config['exp_name']}"

    # Accelerator
    accelerator = Accelerator(project_dir=exp_dir, log_with="tensorboard")
    device = accelerator.device

    if accelerator.is_local_main_process:
        os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

        # add the dataset_config to the main config
        with open(
            f"{config['dataset_base_dir']}/{config['dataset_name']}/config.yaml", "r"
        ) as f:
            dataset_config = yaml.load(f, Loader=yaml.FullLoader)
        config.update(dataset_config)

        # save the config file
        with open(f"{exp_dir}/config.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)

    ### Train dataset with dataloader
    train_dataset = DatasetDDPM(
        f"{config['dataset_base_dir']}/{config['dataset_name']}"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    ### Model
    model = ConditionalUnet(config["ddpm_params"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=len(train_dataloader) // 2,
        num_training_steps=config["n_epochs"] * len(train_dataloader),
    )

    # avoid bucket view strides error
    model.to(memory_format=torch.channels_last)

    ####################################################

    model, opt, train_dataloader, lr_scheduler = accelerator.prepare(
        model, opt, train_dataloader, lr_scheduler
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True
    )
    loss_fn = torch.nn.MSELoss()

    accelerator.init_trackers(config["exp_name"])

    ### Training
    curr_iter = 0
    for epoch in range(config["n_epochs"]):
        
        for x, y in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            mininterval=5,
        ):
            # sample the noise and the timesteps
            noise = torch.randn_like(x)

            timesteps = (
                torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],))
                .long()
                .to(accelerator.device)
            )

            # Add the noise to the input
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get the model prediction
            pred = model(sample=noisy_x, timestep=timesteps, conditioning=y).sample

            # Calculate the loss
            loss = loss_fn(pred, noise)  # How close is the output to the noise

            # Backprop and update the params:
            opt.zero_grad()
            accelerator.backward(loss)

            opt.step()
            lr_scheduler.step()
            
            # log the loss every 10% of the dataset
            if curr_iter % (len(train_dataloader) // 10) == 0:
                accelerator.log({"loss/train": loss.item()}, step=curr_iter)
            
            curr_iter += 1
            
        if accelerator.is_local_main_process:
            logging.info(f"Epoch {epoch}, loss: {loss.item():.4f}")
            
        # save the model checkpoint
        if epoch > 0 and epoch % config["checkpoint_every"] == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, f"{exp_dir}/checkpoints/epoch_{epoch}")

    accelerator.wait_for_everyone()
    accelerator.save_model(model, f"{exp_dir}/checkpoints/epoch_{epoch}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--dataset_name", type=str)

    parser.add_argument("--sample_size", type=int)

    parser.add_argument("--block_out_channels", type=str)

    args = parser.parse_args()

    run(args)
