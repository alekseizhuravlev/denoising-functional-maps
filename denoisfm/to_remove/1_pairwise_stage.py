import argparse
import os
import random

import DenoisingFunctionalMaps.denoisfm.map_selection as map_selection
import denoisfm.utils.fmap_util as fmap_util
import denoisfm.utils.geodist_util as geodist_util
import denoisfm.utils.preprocessing_util as preprocessing_util
import denoisfm.utils.shape_util as shape_util
import numpy as np
import torch
import trimesh
import yaml


def template_map_to_pairwise_map(
    p2p_1, p2p_2,
    evecs_1, evecs_2,
    num_evecs_init,
    mass_2=None
    ):
        
    C_12 = torch.linalg.lstsq(
        evecs_2[:, :num_evecs_init][p2p_2],
        evecs_1[:, :num_evecs_init][p2p_1]
        ).solution
    
    C_12 = fmap_util.zoomout(
        FM_12=C_12, 
        evects1=evecs_1,
        evects2=evecs_2,
        nit=evecs_1.shape[1] - num_evecs_init, step=1,
        A2=mass_2
    )

    p2p = fmap_util.fmap2pointmap(
        C12=C_12,
        evecs_x=evecs_1,
        evecs_y=evecs_2,
        ).cpu()
    
    return p2p



def pairwise_stage(
    test_shape_1, test_shape_2,
    p2p_template_1, p2p_template_2,
    gt_corr_1, gt_corr_2,
    config,
    num_samples_median
):
       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evecs_1 = test_shape_1['evecs'].to(device)
    evecs_2 = test_shape_2['evecs'].to(device)
    
    mass_2 = test_shape_2['mass'].to(device)
    
    p2p_template_1 = p2p_template_1.to(device)
    p2p_template_2 = p2p_template_2.to(device)
    
    ###############################################
    # Map refinement with ZoomOut
    ###############################################
    
    p2p_pair_list = []
    for k in range(p2p_template_1.shape[0]):
        
        p2p_pair_k = template_map_to_pairwise_map(
            p2p_template_1[k], p2p_template_2[k],
            evecs_1, evecs_2,
            num_evecs_init=config["model_params"]["sample_size"],
            mass_2=mass_2
            )
        p2p_pair_list.append(p2p_pair_k)
        
    p2p_pair_list = torch.stack(p2p_pair_list)
    
    ###############################################
    # Geodesic distance matrices
    ###############################################
    
    print('Computing geodesic distance matrix...')
    
    dist_1 = torch.tensor(shape_util.compute_geodesic_distmat(
        test_shape_1['verts'].numpy(),
        test_shape_1['faces'].numpy(),    
    ))
    
    
    ###############################################
    # Median map selection
    ###############################################
    
    p2p_median = map_selection.select_p2p_map(
        p2p_pair_list,
        test_shape_1['verts'],
        test_shape_2['L'], 
        dist_1,
        num_samples_median=num_samples_median
        ).cpu()
    
    geo_err = geodist_util.calculate_geodesic_error(
        dist_1, gt_corr_1.cpu(), gt_corr_2.cpu(), p2p_median, return_mean=True
    ) * 100
    
    print(f'Geodesic error: {geo_err:.2f}')
    
    return p2p_median



def run(args):
    
    # set random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    
    #######################################################
    # Configuration
    #######################################################
    
    experiment_name = args.experiment_name

    ### config
    exp_base_folder = f'checkpoints/ddpm/{experiment_name}'
    with open(f'{exp_base_folder}/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
   
    ###################################################
    # Load the test shapes
    ###################################################
    
    test_shape_1 = trimesh.load(
        args.test_shape_1,
        process=False
    )
    test_shape_1 = preprocessing_util.preprocessing_pipeline(
        test_shape_1.vertices, test_shape_1.faces,
        num_evecs=200, centering='bbox'
        )    
    
    test_shape_2 = trimesh.load(
        args.test_shape_2,
        process=False
    )
    test_shape_2 = preprocessing_util.preprocessing_pipeline(
        test_shape_2.vertices, test_shape_2.faces,
        num_evecs=200, centering='bbox'
        )
    
    p2p_template_1 = torch.load(args.p2p_template_1, weights_only=True)
    p2p_template_2 = torch.load(args.p2p_template_2, weights_only=True)
    
    # load ground truth p2p maps as txt, then convert to torch tensor
    # subtract 1 to make the indices 0-based
    gt_corr_1 = torch.tensor(
        np.loadtxt(args.gt_corr_1)
        ).long() - 1
    gt_corr_2 = torch.tensor(
        np.loadtxt(args.gt_corr_2)
        ).long() - 1
    

    p2p_pred = pairwise_stage(
        test_shape_1, test_shape_2,
        p2p_template_1, p2p_template_2,
        gt_corr_1, gt_corr_2,
        config,
        num_samples_median=args.num_samples_median
    )
    
    # save the output
    test_shape_name_1 = os.path.basename(args.test_shape_1).replace('.off', '')
    test_shape_name_2 = os.path.basename(args.test_shape_2).replace('.off', '')
        
    torch.save(p2p_pred, f'results/pairwise_stage_{test_shape_name_1}_{test_shape_name_2}.pt')
    
     
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test the model')
    
    parser.add_argument('--experiment_name', type=str, required=True)
    
    parser.add_argument('--test_shape_1', type=str, required=True)
    parser.add_argument('--test_shape_2', type=str, required=True)
    
    parser.add_argument('--p2p_template_1', type=str, required=True)
    parser.add_argument('--p2p_template_2', type=str, required=True)
    
    parser.add_argument('--gt_corr_1', type=str, required=False)
    parser.add_argument('--gt_corr_2', type=str, required=False)
        
    parser.add_argument('--num_samples_median', type=int, default=16)
    
    args = parser.parse_args()
    

    run(args)
