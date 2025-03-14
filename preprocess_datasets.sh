dataset_choices=(
    # "FAUST_r" "SCAPE_r" "SHREC19_r"
    # "FAUST_a" "SCAPE_a"
    "DT4D_r"
    "SMAL_r"
)
base_dir="/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm"
for dataset in "${dataset_choices[@]}"
do
    echo "Processing ${dataset}"
    python preprocess.py --data_root "${base_dir}/${dataset}"
done