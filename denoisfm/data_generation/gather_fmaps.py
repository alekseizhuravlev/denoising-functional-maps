import numpy as np
import os
import torch
import time

def get_files_with_prefix(files, prefix):
    
    files_with_start_end_idxs = []
    for file in files:
        if prefix in file:
            file_no_prefix = file.replace(f'{prefix}_', '')
            # remove extension
            file_no_prefix = file_no_prefix.split('.')[0]
            
            # get the start and end indices
            start_idx, end_idx = file_no_prefix.split('_')
            
            files_with_start_end_idxs.append({
                    'file': file,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx)
            })
            
    sorted_files = sorted(files_with_start_end_idxs, key=lambda x: x['start_idx'])
    
    return sorted_files


def verify_integrity(data_dir, prefix_list):
    
    # check that for each prefix, all files have identical start and end indices
    
    files = os.listdir(data_dir)
    files = sorted(files)
    
    files_by_prefix = {}
    unique_start_indices = set()
    
    for prefix in prefix_list:
        
        sorted_files = get_files_with_prefix(files, prefix)
        
        files_by_prefix[prefix] = sorted_files
        
        unique_start_indices.update([file['start_idx'] for file in sorted_files])
        

    # assert that for each prefix, there exists one file with each start index
    for prefix in prefix_list:
        start_indices = [file['start_idx'] for file in files_by_prefix[prefix]]
        assert len(start_indices) == len(unique_start_indices), f'prefix: {prefix}, len(start_indices): {len(start_indices)}, len(unique_start_indices): {len(unique_start_indices)}'
        
        print(f'prefix: {prefix}, len(start_indices): {len(start_indices)}, len(unique_start_indices): {len(unique_start_indices)}')
        
        assert set(start_indices) == unique_start_indices, f'prefix: {prefix}, start_indices: {start_indices}, unique_start_indices: {unique_start_indices}'
            
        print(f'prefix: {prefix}, start_indices: {start_indices}, unique_start_indices: {sorted(unique_start_indices)}')
    
    
def check_for_nan(data_dir, prefix_list):
    
    # check that for each prefix, all files have identical start and end indices
    
    files = os.listdir(data_dir)
    files = sorted(files)
    
    files_by_prefix = {}
    unique_start_indices = set()
    
    for prefix in prefix_list:
        
        sorted_files = get_files_with_prefix(files, prefix)
        
        files_by_prefix[prefix] = sorted_files
        

    # assert that for each prefix, there exists one file with each start index
    for prefix in prefix_list:
        # load each file and check for nan
        for file in files_by_prefix[prefix]:
            data = torch.load(f'{data_dir}/{file["file"]}')
            
            assert not torch.isnan(data).any(), f'prefix: {prefix}, file: {file["file"]} has nan'
        
        print(f'prefix: {prefix}, no nans found')   
    


def gather_files(data_dir, prefix, remove_after):
    
    # if f'{data_dir}/{prefix}.txt' exists, remove it
    # if os.path.exists(f'{data_dir}/{prefix}.txt'):
    #     print('Removing', f'{data_dir}/{prefix}.txt')
    #     os.remove(f'{data_dir}/{prefix}.txt')
    
    if os.path.exists(f'{data_dir}/{prefix}.pt'):
        raise RuntimeError(f'{data_dir}/{prefix}.pt already exists')
        # print('Removing', f'{data_dir}/{prefix}.pt')
        # os.remove(f'{data_dir}/{prefix}.pt')
    
    # get all files in dir in alphabetical order
    files = os.listdir(data_dir)
    files = sorted(files)
    
    sorted_files = get_files_with_prefix(files, prefix)
    
    data_pt = torch.tensor([])

    time_start = time.time()

    for file in sorted_files:
        # read the file as numpy array
        # data_i = np.loadtxt(f'{data_dir}/{file["file"]}')
        # data_i = torch.tensor(data_i)
        
        # sleep for 1-2 seconds
        # time.sleep(np.random.uniform(1, 2))
        
        # read the file as torch tensor
        data_i = torch.load(f'{data_dir}/{file["file"]}')
        assert not torch.isnan(data_i).any(), f'file: {file["file"]} has nan'
        
        # raise RuntimeError('this should be a stack')
        data_pt = torch.cat((data_pt, data_i), dim=0)
        
        time_end = time.time()
        print(f'{time_end - time_start:.2f}: Appending', file['file'], 'shape:', data_i.shape, 'total shape:', data_pt.shape)
        time_start = time_end
            
    # save the data to a .pt file
    torch.save(data_pt, f'{data_dir}/{prefix}.pt')
    
    # remove the files
    if remove_after:
        for file in sorted_files:
            print('Removing', f'{file["file"]}')
            os.remove(f'{data_dir}/{file["file"]}')
            
         
   
    

if __name__ == '__main__':
    
    # dataset_name = 'SURREAL_96_1-2-2ev_template_remeshed_augShapes'
    
    dataset_name_list = [
        # 'SURREAL_sign_net_32_norm_rm',
        # 'SURREAL_sign_net_64_norm_rm',
        # 'SURREAL_sign_net_96_122_norm_rm',
        # 'SURREAL_sign_net_96_124_norm_rm',
        
        'SMAL_sign_net_64_smal',
        'SMAL_sign_net_64_smal_old_data',
        
        # 'SMAL_nocat_64_SMAL_isoRemesh_0.2_0.8_nocat_1-2ev_32k',
        # 'SMAL_nocat_64_SMAL_isoRemesh_0.2_0.8_nocat_1-2ev_64k',
    ]
    
    prefix_list = [
        # 'evals_first', 'evals_second',
        # 'C_gt_xy',
        'C_1T', 
        'y_1', 'y_T'
        ]
    
    for dataset_name in dataset_name_list:
        
        print('Gathering', dataset_name)
        time.sleep(2)
        
        # data_dir = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL/train/{dataset_name}/train'
        data_dir = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/{dataset_name}'
        
        verify_integrity(data_dir, prefix_list)
        
        # check_for_nan(data_dir, prefix_list)
        
        
        for prefix in prefix_list:
            gather_files(data_dir, prefix, remove_after=True)

    
    