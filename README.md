# Denoising Functional Maps: Diffusion Models for Shape Correspondence

This code provides an example for inference on a test pair of meshes.

## Installation
```bash 
conda create -n denoisfm python=3.8
conda activate denoisfm
pip install -r requirements.txt --no-cache-dir
```

## Test
Run the inference of a pretrained model on a test pair of meshes.
The inference consists of two stages:
- Template stage, run once for each unique mesh in the dataset
- Pairwise stage, run for each test pair

```python
cd path/to/project
export PYTHONPATH='path/to/project'

# first mesh
python 0_template_stage.py \
    --experiment_name ddpm_64 --checkpoint_name epoch_99 \
    --test_shape data/example/off/tr_reg_082.off \
    --num_iters_avg 16

# second mesh
python 0_template_stage.py \
    --experiment_name ddpm_64 --checkpoint_name epoch_99 \
    --test_shape data/example/off/tr_reg_096.off \
    --num_iters_avg 16

# pairwise
python 1_pairwise_stage.py --experiment_name ddpm_64 \
    --test_shape_1 data/example/off/tr_reg_082.off \
    --test_shape_2 data/example/off/tr_reg_096.off \
    --p2p_template_1 results/template_stage_tr_reg_082.pt \
    --p2p_template_2 results/template_stage_tr_reg_096.pt \
    --gt_corr_1 data/example/corres/tr_reg_082.vts \
    --gt_corr_2 data/example/corres/tr_reg_096.vts \
    --num_samples_median 4
```

The template-wise and pair-wise maps will be saved in [results](results) folder.

For visualization, check the `visualize_correspondence.ipynb`