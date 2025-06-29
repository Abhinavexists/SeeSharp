import os
import sys
from dataset import VimeoDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

def show_dataset_info(dataset_path):
    """Show detailed information about the dataset structure"""
    print(f"\n=== Dataset Structure Analysis ===")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path {dataset_path} does not exist!")
        return
    
    print(f"Dataset root: {dataset_path}")
    root_contents = os.listdir(dataset_path)
    print(f"Root contents: {root_contents}")
    
    vimeo_dir = os.path.join(dataset_path, 'vimeo_settuplet_1')
    if os.path.exists(vimeo_dir) and os.path.isdir(vimeo_dir):
        print(f"Found vimeo_settuplet_1 directory")
        print(f"Contents of vimeo_settuplet_1: {os.listdir(vimeo_dir)[:10]}")
        
        sequences_dir = os.path.join(vimeo_dir, 'sequences')
        if os.path.exists(sequences_dir) and os.path.isdir(sequences_dir):
            print(f"Found sequences directory at {sequences_dir}")
            seq_folders = [d for d in os.listdir(sequences_dir) if os.path.isdir(os.path.join(sequences_dir, d))]
            print(f"Found {len(seq_folders)} sequence folders in sequences directory")
            if seq_folders:
                print(f"Example folders: {seq_folders[:5]}")
                
                sample_folder = os.path.join(sequences_dir, seq_folders[0])
                print(f"Contents of {seq_folders[0]}: {os.listdir(sample_folder)[:10]}")
                
                subdirs = [d for d in os.listdir(sample_folder) if os.path.isdir(os.path.join(sample_folder, d))]
                if subdirs:
                    print(f"Found {len(subdirs)} subdirectories in {seq_folders[0]}")
                    sample_subdir = os.path.join(sample_folder, subdirs[0])
                    print(f"Contents of {seq_folders[0]}/{subdirs[0]}: {os.listdir(sample_subdir)}")
        
        else:
            print(f"No sequences directory found, checking vimeo_settuplet_1 directly")
            image_files = [f for f in os.listdir(vimeo_dir) if f.endswith('.png')]
            if image_files:
                print(f"Found {len(image_files)} image files in vimeo_settuplet_1, e.g.: {image_files[:5]}")
            
            subdirs = [d for d in os.listdir(vimeo_dir) if os.path.isdir(os.path.join(vimeo_dir, d))]
            if subdirs:
                print(f"Found directories in vimeo_settuplet_1: {subdirs[:10]}")
                
                sample_dir = os.path.join(vimeo_dir, subdirs[0])
                print(f"Contents of {subdirs[0]}: {os.listdir(sample_dir)[:10]}")
    
    else:
        for dirname in ['sequence', 'sequences']:
            seq_dir = os.path.join(dataset_path, dirname)
            if os.path.exists(seq_dir):
                print(f"Found {dirname} directory at {seq_dir}")
                
                seq_contents = os.listdir(seq_dir)
                print(f"Contents of {dirname}: {seq_contents[:10]}")
                break
    
    for list_name in ['sep_trainlist.txt', 'sep_testlist.txt', 'test.txt']:
        list_path = os.path.join(dataset_path, list_name)
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                lines = f.readlines()
                print(f"Found {list_name} with {len(lines)} entries")
                if lines:
                    sample_entries = [line.strip() for line in lines[:3]]
                    print(f"Sample entries: {sample_entries}")
        else:
            print(f"Could not find {list_name}")
    
    print("=== End of Dataset Analysis ===\n")

def visualize_sample(dataset):
    """Visualize a random sample from the dataset"""
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    idx = random.randint(0, len(dataset) - 1)
    lr_frames, hr_frame = dataset[idx]
    
    print(f"Sample shapes:")
    print(f"  LR frames: {lr_frames.shape}")
    print(f"  HR frame: {hr_frame.shape}")
    
    lr_frames = lr_frames.numpy()
    hr_frame = hr_frame.numpy()
    
    lr_vis = np.transpose(lr_frames, (1, 2, 3, 0))
    hr_vis = np.transpose(hr_frame, (1, 2, 0))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i in range(3):
        axes[i].imshow(lr_vis[i])
        axes[i].set_title(f"LR Frame {i+1}")
        axes[i].axis('off')
    
    axes[3].imshow(hr_vis)
    axes[3].set_title(f"HR Frame (Target)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    print("Sample visualization saved to 'sample_visualization.png'")

def test_data_loading(dataset_path, split_list=None, batch_size=2):
    """Test loading the dataset with DataLoader"""
    try:
        if split_list and os.path.exists(split_list):
            print(f"Creating dataset with split list: {split_list}")
            dataset = VimeoDataset(dataset_path, split_list=split_list)
        else:
            print("Creating dataset without split list")
            dataset = VimeoDataset(dataset_path)
        
        print(f"Dataset size: {len(dataset)} sequences")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        print(f"Testing dataloader with {len(dataloader)} batches")
        
        num_batches_to_test = min(5, len(dataloader))
        for i, (lr_frames, hr_frames) in enumerate(dataloader):
            if i >= num_batches_to_test:
                break
                
            print(f"Batch {i+1}:")
            print(f"  LR frames shape: {lr_frames.shape}")
            print(f"  HR frames shape: {hr_frames.shape}")
            print(f"  LR frames range: [{lr_frames.min():.4f}, {lr_frames.max():.4f}]")
            print(f"  HR frames range: [{hr_frames.min():.4f}, {hr_frames.max():.4f}]")
        
        visualize_sample(dataset)
        
        print("\n✅ Dataset loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during dataset loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    dataset_path = 'archive'
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    show_dataset_info(dataset_path)
    
    train_list = os.path.join(dataset_path, 'sep_trainlist.txt')
    if os.path.exists(train_list):
        print("\nTesting with training split list...")
        test_data_loading(dataset_path, train_list)
    else:
        print("\nTesting without split list...")
        test_data_loading(dataset_path) 