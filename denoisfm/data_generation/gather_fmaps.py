import os
import time

import torch


def get_files_with_prefix(files, prefix):
    """
    Filters and sorts files in a directory that start with a specified prefix and extracts their start and end indices.

    Args:
        files (list of str): List of filenames in the directory.
        prefix (str): The prefix that the filenames should start with.

    Returns:
        list of dict: A sorted list of dictionaries containing "file", "start_idx", and "end_idx" keys.
    """

    files_with_start_end_idxs = []
    for file in files:
        if prefix in file:
            file_no_prefix = file.replace(f"{prefix}_", "")
            # remove extension
            file_no_prefix = file_no_prefix.split(".")[0]

            # get the start and end indices
            start_idx, end_idx = file_no_prefix.split("_")

            files_with_start_end_idxs.append(
                {"file": file, "start_idx": int(start_idx), "end_idx": int(end_idx)}
            )

    sorted_files = sorted(files_with_start_end_idxs, key=lambda x: x["start_idx"])

    return sorted_files


def verify_integrity(data_dir, prefix_list):
    """
    Checks that each prefix has files with matching start and end indices,
    ensuring that no start index is missing for any given prefix.

    Args:
        data_dir (str): The directory containing the files.
        prefix_list (list of str): List of prefixes to check for corresponding files in the directory.

    Raises:
        AssertionError: If there are any missing or mismatched start indices for any of the prefixes.
    """

    files = os.listdir(data_dir)
    files = sorted(files)

    files_by_prefix = {}
    unique_start_indices = set()

    for prefix in prefix_list:
        sorted_files = get_files_with_prefix(files, prefix)

        files_by_prefix[prefix] = sorted_files

        unique_start_indices.update([file["start_idx"] for file in sorted_files])

    # assert that for each prefix, there exists one file with each start index
    for prefix in prefix_list:
        start_indices = [file["start_idx"] for file in files_by_prefix[prefix]]
        assert len(start_indices) == len(unique_start_indices), (
            f"prefix: {prefix}, len(start_indices): {len(start_indices)}, len(unique_start_indices): {len(unique_start_indices)}"
        )
        assert set(start_indices) == unique_start_indices, (
            f"prefix: {prefix}, start_indices: {start_indices}, unique_start_indices: {unique_start_indices}"
        )

        print(f"prefix: {prefix}, start_indices: {start_indices}")


def gather_files(data_dir, prefix, remove_after):
    """
    Gathers and concatenates tensor data from multiple files matching a given prefix,
    and saves the concatenated result to a new file.

    Args:
        data_dir (str): The directory containing the files to be gathered.
        prefix (str): The prefix to identify the files to gather.
        remove_after (bool): Whether to remove the original files after they are gathered and concatenated.
    """
    if os.path.exists(f"{data_dir}/{prefix}.pt"):
        raise RuntimeError(f"{data_dir}/{prefix}.pt already exists")

    # get all files in dir in alphabetical order
    files = os.listdir(data_dir)
    files = sorted(files)

    sorted_files = get_files_with_prefix(files, prefix)

    data_pt = torch.tensor([])

    time_start = time.time()

    for file in sorted_files:
        # read the file as torch tensor
        data_i = torch.load(f"{data_dir}/{file['file']}")

        assert not torch.isnan(data_i).any(), f"file: {file['file']} has nan"
        assert not torch.isinf(data_i).any(), f"file: {file['file']} has inf"

        data_pt = torch.cat((data_pt, data_i), dim=0)

        time_end = time.time()
        print(
            f"{time_end - time_start:.2f}: Appending {file['file']} shape: {data_i.shape} total shape: {data_pt.shape}"
        )
        time_start = time_end

    # save the data to a .pt file
    torch.save(data_pt, f"{data_dir}/{prefix}.pt")

    # remove the appended files
    if remove_after:
        for file in sorted_files:
            print("Removing", f"{file['file']}")
            os.remove(f"{data_dir}/{file['file']}")


if __name__ == "__main__":
    dataset_name_list = [
        "SMAL_128000_sign_net_64_smal_new_proj",
        "SMAL_192000_sign_net_64_smal_new_proj",
        "SMAL_sign_net_64_smal_new_proj",
    ]

    prefix_list = ["C_1T", "y_1", "y_T"]

    for dataset_name in dataset_name_list:
        print("Gathering", dataset_name)
        time.sleep(2)

        data_dir = f"/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/ddpm/{dataset_name}"

        verify_integrity(data_dir, prefix_list)

        for prefix in prefix_list:
            gather_files(data_dir, prefix, remove_after=False)
