import torch
import numpy as np
import os
from tqdm import tqdm
import yaml
import argparse
import accelerate
import random
import trimesh

# models
from diffusers import DDPMScheduler
from models.conditional_unet import ConditionalUnet
import models.feature_extractor as feature_extractor

from sign_correction.inference import predict_sign_change

import utils.fmap_util as fmap_util
import utils.preprocessing_util as preprocessing_util



def template_stage(
        model, sign_corr_net, noise_scheduler,
        test_shape, template_shape,
        config, num_iters_avg
    ):
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_evecs = config["model_params"]["sample_size"]
    
    
    ###############################################
    # !!! in the following, template -> 1, test -> 2
    ###############################################
    
    verts_1 = template_shape['verts'].unsqueeze(0).to(device)
    verts_2 = test_shape['verts'].unsqueeze(0).to(device)
    
    faces_1 = template_shape['faces'].unsqueeze(0).to(device)
    faces_2 = test_shape['faces'].unsqueeze(0).to(device)

    evecs_1 = template_shape['evecs'][:, :num_evecs].unsqueeze(0).to(device)
    evecs_2 = test_shape['evecs'][:, :num_evecs].unsqueeze(0).to(device)
    

    mass_mat_1 = torch.diag_embed(
        template_shape['mass'].unsqueeze(0)
        ).to(device)
    mass_mat_2 = torch.diag_embed(
        test_shape['mass'].unsqueeze(0)
        ).to(device)


    ###############################################
    # get conditioning and signs num_iters_avg times
    #
    # (signs instead of evecs because of possible large number of vertices)
    ###############################################

    cond_1_list = []
    cond_2_list = []
    signs_1_list = []
    signs_2_list = []

    for _ in range(num_iters_avg):

        # predict the sign change
        with torch.no_grad():
            projection_values_1, correc_vector_1 = predict_sign_change(
                sign_corr_net,
                verts_1, faces_1,
                evecs_1, 
                mass_mat=mass_mat_1, input_type=sign_corr_net.input_type,
                evecs_per_correc=config["sign_net"]["evecs_per_correc"],
                
                # spectral params for the feature extractor
                mass=template_shape['mass'].unsqueeze(0), L=template_shape['L'].unsqueeze(0),
                evals=template_shape['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=template_shape['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=template_shape['gradX'].unsqueeze(0), gradY=template_shape['gradY'].unsqueeze(0)
                )
            projection_values_2, correc_vector_2 = predict_sign_change(
                sign_corr_net,
                verts_2, faces_2,
                evecs_2, 
                mass_mat=mass_mat_2, input_type=sign_corr_net.input_type,
                evecs_per_correc=config["sign_net"]["evecs_per_correc"],
                
                # spectral params for the feature extractor
                mass=test_shape['mass'].unsqueeze(0), L=test_shape['L'].unsqueeze(0),
                evals=test_shape['evals'][:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                evecs=test_shape['evecs'][:,:config["sign_net"]["net_params"]["k_eig"]].unsqueeze(0),
                gradX=test_shape['gradX'].unsqueeze(0), gradY=test_shape['gradY'].unsqueeze(0)
                )

        # correct the evecs
        evecs_1_corrected = evecs_1[0] * torch.sign(projection_values_1)
        evecs_1_corrected_norm = evecs_1_corrected / torch.norm(evecs_1_corrected, dim=0, keepdim=True)
        
        evecs_2_corrected = evecs_2[0] * torch.sign(projection_values_2)
        evecs_2_corrected_norm = evecs_2_corrected / torch.norm(evecs_2_corrected, dim=0, keepdim=True)
        

        # mass matrices
        mass_mat_1 = torch.diag_embed(
            template_shape['mass'].unsqueeze(0)
            ).to(device)
        mass_mat_2 = torch.diag_embed(
            test_shape['mass'].unsqueeze(0)
            ).to(device)
        
        # conditioning = correction_vector @ mass @ corrected evecs
        
        cond_1 = torch.nn.functional.normalize(
            correc_vector_1[0].transpose(0, 1) \
                @ mass_mat_1[0],
            p=2, dim=1) \
                @ evecs_1_corrected_norm
        
        cond_2 = torch.nn.functional.normalize(
            correc_vector_2[0].transpose(0, 1) \
                @ mass_mat_2[0],
            p=2, dim=1) \
                @ evecs_2_corrected_norm 
               
        cond_1_list.append(cond_1)
        cond_2_list.append(cond_2)
        signs_1_list.append(torch.sign(projection_values_1))
        signs_2_list.append(torch.sign(projection_values_2))
        
    cond_1_list = torch.stack(cond_1_list)
    cond_2_list = torch.stack(cond_2_list)
    signs_1_list = torch.stack(signs_1_list)
    signs_2_list = torch.stack(signs_2_list)    
    
    
    ###############################################
    # Conditioning
    ###############################################

    conditioning = torch.cat(
        (cond_1_list.unsqueeze(1), cond_2_list.unsqueeze(1)),
        1)
    
    ###############################################
    # Forward diffusion process
    ###############################################
    
    x_sampled = torch.rand(num_iters_avg, 1, 
                        config["model_params"]["sample_size"],
                        config["model_params"]["sample_size"]).to(device)
    y = conditioning.to(device)    
        
    # Sampling loop
    for t in tqdm(noise_scheduler.timesteps, desc='Denoising...'):

        # Get model pred
        with torch.no_grad():
            residual = model(x_sampled, t,
                                conditioning=y
                                ).sample

        # Update sample with step
        x_sampled = noise_scheduler.step(residual, t, x_sampled).prev_sample
    
        
    C_21_est_list = x_sampled
    
    ##########################################################
    # Convert fmaps to p2p maps to template
    ##########################################################
    
    p2p_est_list = []
    
    for k in range(num_iters_avg):

        evecs_1_corrected = evecs_1[0,:, :num_evecs] * signs_1_list[k]
        evecs_2_corrected = evecs_2[0,:, :num_evecs] * signs_2_list[k]
        Cyx_est_k = C_21_est_list[k][0]

        p2p_est_k = fmap_util.fmap2pointmap(
            C12=Cyx_est_k,
            evecs_x=evecs_2_corrected,
            evecs_y=evecs_1_corrected,
            ).cpu()

        p2p_est_list.append(p2p_est_k)
    
    p2p_est_list = torch.stack(p2p_est_list)
          
        
    return p2p_est_list




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
    exp_base_folder = f'checkpoints/ddpm/{experiment_name}'
    with open(f'{exp_base_folder}/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    #######################################################
    # Model setup
    #######################################################

    ### DDPM model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConditionalUnet(config["model_params"])

    if "accelerate" in config and config["accelerate"]:
        accelerate.load_checkpoint_in_model(model, f"{exp_base_folder}/checkpoints/{checkpoint_name}/model.safetensors")
    else:
        model.load_state_dict(torch.load(f"{exp_base_folder}/checkpoints/{checkpoint_name}"))
    
    model.to(device)
    
    ### Sign correction network
    sign_corr_net = feature_extractor.DiffusionNet(
        **config["sign_net"]["net_params"]
        )        
    sign_corr_net.load_state_dict(torch.load(
            f'checkpoints/sign_net/{config["sign_net"]["net_name"]}/{config["sign_net"]["n_iter"]}.pth'
            ))
    sign_corr_net.to(device)

    ### noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True) 
    
    
    ##########################################
    # Template shape
    ##########################################

    template_shape = trimesh.load(
        f'data/template/template.off',
        process=False
    )  
    template_shape = preprocessing_util.preprocessing_pipeline(
        template_shape.vertices, template_shape.faces,
        num_evecs=128, centering='bbox'
        ) 

    ##########################################
    # Test shape
    ##########################################
    
    test_shape = trimesh.load(
        args.test_shape_path,
        process=False
    )
    test_shape = preprocessing_util.preprocessing_pipeline(
        test_shape.vertices, test_shape.faces,
        num_evecs=128, centering='bbox'
        )
    
    ##########################################
    # Template stage
    ##########################################
    
    p2p_maps_template = template_stage(
        model, sign_corr_net, noise_scheduler,
        test_shape, template_shape,
        config, num_iters_avg=args.num_iters_avg
    )
    
    ##########################################
    # Save the template-wise maps
    ##########################################
    
    # get name of test shape
    test_shape_name = os.path.basename(args.test_shape_path).replace('.off', '')
    save_path = f'results/template_stage_{test_shape_name}.pt'
    
    torch.save(p2p_maps_template, save_path)
    
    
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test the model')
    
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    
    parser.add_argument('--test_shape_path', type=str, required=True)
    
    parser.add_argument('--num_iters_avg', type=int, default=128)


    args = parser.parse_args()
    
    
    # python 0_template_stage.py --experiment_name ddpm_64 --checkpoint_name epoch_99 --test_shape_path data/example/off/tr_reg_082.off

    run(args)
