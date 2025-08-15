import os

import torch


def setup_device(gpu_id=None):
    """Setup the computing device"""
    if torch.cuda.is_available():
        print("Available GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using specified GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        elif torch.cuda.device_count() > 1:
            torch.cuda.set_device(
                1
            )  # Use GPU 1 if available and no specific GPU specified
            device = torch.device("cuda:1")
            print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            device = torch.device("cuda:0")
            print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")

        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    return device


def check_dataset_structure(data_path):
    """Check and analyze the dataset structure"""
    print("\n--- Dataset Structure Analysis ---")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} does not exist!")
        return

    print(f"Contents of {data_path}:")
    data_contents = os.listdir(data_path)
    print(f"Found {len(data_contents)} items: {data_contents[:10]}...")

    sequence_path = os.path.join(data_path, "sequences")
    if os.path.exists(sequence_path):
        print(f"Found sequence folder at {sequence_path}")

        # Count sequence folders
        sequence_dirs = [
            d
            for d in os.listdir(sequence_path)
            if os.path.isdir(os.path.join(sequence_path, d))
        ]
        print(f"Found {len(sequence_dirs)} sequence directories")

        if sequence_dirs:
            # Check first sequence folder
            first_seq = sequence_dirs[0]
            first_seq_path = os.path.join(sequence_path, first_seq)
            subseqs = [
                d
                for d in os.listdir(first_seq_path)
                if os.path.isdir(os.path.join(first_seq_path, d))
            ]
            print(f"Sequence {first_seq} contains {len(subseqs)} sub-sequences")

            if subseqs:
                # Check first sub-sequence
                first_subseq = subseqs[0]
                first_subseq_path = os.path.join(first_seq_path, first_subseq)
                files = os.listdir(first_subseq_path)
                print(
                    f"Sub-sequence {first_seq}/{first_subseq} contains files: {files}"
                )

    train_list = os.path.join(data_path, "sep_trainlist.txt")
    test_list = os.path.join(data_path, "sep_testlist.txt")

    if os.path.exists(train_list):
        with open(train_list) as f:
            lines = f.readlines()
            print(f"Found sep_trainlist.txt with {len(lines)} entries")
            if lines:
                print(f"First 3 entries: {[line.strip() for line in lines[:3]]}")
    else:
        print(f"WARNING: {train_list} not found")

    if os.path.exists(test_list):
        with open(test_list) as f:
            lines = f.readlines()
            print(f"Found sep_testlist.txt with {len(lines)} entries")
    else:
        print(f"WARNING: {test_list} not found")

    print("--- End of Dataset Analysis ---\n")