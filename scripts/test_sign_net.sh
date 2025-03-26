dataset_choices=(
    "FAUST_r" "SCAPE_r" "SHREC19_r"
    "FAUST_a" "SCAPE_a"
)
checkpoint_name="50000.pth"
exp_name=sign_net_32

echo "Testing sign correction accuracy for ${exp_name} with checkpoint ${checkpoint_name}"

data_dir="/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/test"
for dataset in "${dataset_choices[@]}"; do
    echo "Processing ${dataset}"
    python test_sign_net.py \
        --exp_name $exp_name \
        --checkpoint_name $checkpoint_name \
        --dataset_name $dataset \
        --data_dir $data_dir \
        --n_epochs 100
done