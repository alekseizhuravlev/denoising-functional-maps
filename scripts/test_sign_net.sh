dataset_choices=(
    "FAUST_r" "SCAPE_r" "SHREC19_r"
    "FAUST_a" "SCAPE_a"
)
checkpoint_names=(
    "50000.pth"
    #  "35000.pth" 
    # "25000.pth" "15000.pth"
)
exp_name=sign_net_96_124_norm_rm_anis

echo $exp_name

base_dir="/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm"
for dataset in "${dataset_choices[@]}"; do
    for checkpoint in "${checkpoint_names[@]}"; do
        echo "Processing ${dataset} with checkpoint ${checkpoint}"
        python test_sign_net.py \
            --exp_name "${exp_name}" \
            --dataset_name "${dataset}" \
            --base_dir "${base_dir}/test" \
            --n_epochs 100 \
            --checkpoint_name "${checkpoint}"
    done
done