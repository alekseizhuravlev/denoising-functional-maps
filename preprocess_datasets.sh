# dataset_choices = [
#     "FAUST_r", "SCAPE_r", "SHREC19_r",
#     "FAUST_a", "SCAPE_a",
#     "DT4D_intra", "DT4D_inter",
#     "SMAL_iso"
# ]
# base_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm
# for each subfolder in base_dir, run the following command:
# python preprocess.py --data_root {base_dir}/{subfolder}
# I wrote it in python, use shell script to run it for each subfolder

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