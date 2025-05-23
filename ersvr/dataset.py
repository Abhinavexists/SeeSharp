import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import random

class VimeoDataset(Dataset):
    def __init__(self, root_dir, split_list=None, sample_size=None, verbose=True):
        self.root_dir = root_dir
        self.sequences = []
        self.verbose = verbose
        
        if verbose:
            print(f"Initializing dataset from {self.root_dir}")
            print(f"Using split list: {split_list if split_list else 'None'}")

        # Check if root directory exists
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        # Find the actual sequence directory
        seq_dir = None
        
        # Check for vimeo_settuplet_1 directory
        vimeo_septuplet = os.path.join(self.root_dir, "vimeo_settuplet_1")
        if os.path.exists(vimeo_septuplet) and os.path.isdir(vimeo_septuplet):
            if verbose:
                print(f"Found vimeo_settuplet_1 directory")
            
            # Check for sequences directory inside vimeo_septuplet
            sequences_dir = os.path.join(vimeo_septuplet, "sequences")
            if os.path.exists(sequences_dir) and os.path.isdir(sequences_dir):
                seq_dir = sequences_dir
                if verbose:
                    print(f"Found sequences directory at {seq_dir}")
            else:
                # Otherwise use vimeo_septuplet as the sequence directory
                seq_dir = vimeo_septuplet
        else:
            # Try to find a "sequence" or "sequences" directory
            for dirname in ["sequence", "sequences"]:
                candidate = os.path.join(self.root_dir, dirname)
                if os.path.exists(candidate) and os.path.isdir(candidate):
                    seq_dir = candidate
                    if verbose:
                        print(f"Found {dirname} directory at {seq_dir}")
                    break
        
        # If no sequence directory found, use the root directory itself
        if seq_dir is None:
            seq_dir = self.root_dir
            if verbose:
                print(f"No specific sequence directory found, using root: {seq_dir}")
        
        # Store the sequence directory for later use
        self.seq_dir = seq_dir
            
        if split_list is not None and os.path.exists(split_list):
            if verbose:
                print(f"Loading sequences from split list {split_list}")
            with open(split_list, 'r') as f:
                lines = f.readlines()
                if verbose:
                    print(f"Found {len(lines)} sequences in split list")
                for line in lines:
                    seq = line.strip()
                    seq_path = os.path.join(seq_dir, seq)
                    if os.path.exists(seq_path) and self._check_sequence_valid(seq_path):
                        self.sequences.append(seq_path)
                    elif verbose:
                        print(f"Skipping invalid sequence: {seq_path}")
        else:
            if verbose:
                print("No valid split list provided, scanning directory structure")
            
            # Try to identify the structure
            content = os.listdir(seq_dir)
            subdirs = [d for d in content if os.path.isdir(os.path.join(seq_dir, d))]
            
            if verbose:
                print(f"Found {len(subdirs)} items in {seq_dir}")
                if subdirs:
                    print(f"First few items: {subdirs[:5]}")
            
            # Check if this is already a valid sequence directory
            if self._check_sequence_valid(seq_dir):
                if verbose:
                    print(f"Found valid sequence at root level: {seq_dir}")
                self.sequences.append(seq_dir)
            else:
                # Check for numbered directories (typical Vimeo-90k structure)
                numeric_dirs = [d for d in subdirs if d.isdigit() or (len(d) >= 5 and d[:5].isdigit())]
                
                if numeric_dirs and verbose:
                    print(f"Found {len(numeric_dirs)} numeric directories (e.g., {numeric_dirs[:5]})")
                
                # If we have numbered directories, this is likely the standard structure
                if numeric_dirs:
                    for dir_name in numeric_dirs:
                        dir_path = os.path.join(seq_dir, dir_name)
                        # Check if this is a valid sequence
                        if self._check_sequence_valid(dir_path):
                            self.sequences.append(dir_path)
                        else:
                            # Check for subdirectories
                            for subdir in os.listdir(dir_path):
                                subdir_path = os.path.join(dir_path, subdir)
                                if os.path.isdir(subdir_path) and self._check_sequence_valid(subdir_path):
                                    self.sequences.append(subdir_path)
                else:
                    # Otherwise, recursively scan all subdirectories up to a certain depth
                    if verbose:
                        print(f"Scanning subdirectories for sequences...")
                    
                    self._scan_for_sequences(seq_dir, depth=0, max_depth=3)
        
        if verbose:
            print(f"Found {len(self.sequences)} valid sequences")
        
        if sample_size is not None and sample_size < len(self.sequences):
            if verbose:
                print(f"Using random subset of {sample_size} sequences")
            self.sequences = random.sample(self.sequences, sample_size)
    
    def _scan_for_sequences(self, directory, depth=0, max_depth=3):
        """Recursively scan for valid sequences up to max_depth"""
        if depth > max_depth:
            return
        
        # Check if current directory is a valid sequence
        if self._check_sequence_valid(directory):
            self.sequences.append(directory)
            return
        
        # Otherwise, check subdirectories
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    self._scan_for_sequences(item_path, depth + 1, max_depth)
        except Exception as e:
            if self.verbose:
                print(f"Error scanning {directory}: {e}")
    
    def _check_sequence_valid(self, seq_path):
        """Check if a sequence contains all required frame files"""
        # Try different file naming patterns
        patterns = [
            # Pattern 1: im1.png, im2.png, im3.png
            [f'im{i}.png' for i in range(1, 4)],
            # Pattern 2: im01.png, im02.png, im03.png
            [f'im{i:02d}.png' for i in range(1, 4)],
            # Pattern 3: frame001.png, frame002.png, frame003.png
            [f'frame{i:03d}.png' for i in range(1, 4)],
            # Pattern 4: 01.png, 02.png, 03.png
            [f'{i:02d}.png' for i in range(1, 4)]
        ]
        
        # Try each pattern
        for pattern in patterns:
            valid = True
            for frame_name in pattern:
                frame_path = os.path.join(seq_path, frame_name)
                if not os.path.isfile(frame_path):
                    valid = False
                    break
            if valid:
                # Store the valid pattern for later use
                self.file_pattern = pattern
                return True
                
        return False

    def __len__(self):
        return max(1, len(self.sequences))
    
    def __getitem__(self, idx):
        # Handle empty dataset
        if len(self.sequences) == 0:
            dummy_lr = torch.zeros(3, 3, 256, 256)
            dummy_hr = torch.zeros(3, 256, 256)
            return dummy_lr, dummy_hr
        
        # Handle index out of bounds
        if idx >= len(self.sequences):
            idx = idx % len(self.sequences)
            
        seq_path = self.sequences[idx]
        frames = []
        
        try:
            # Determine the file pattern if not already known
            if not hasattr(self, 'file_pattern'):
                self._check_sequence_valid(seq_path)
            
            # If still no file pattern, try common patterns
            if not hasattr(self, 'file_pattern'):
                # Default to im1.png pattern
                self.file_pattern = [f'im{i}.png' for i in range(1, 4)]
            
            # Load frames using the file pattern
            for frame_name in self.file_pattern:
                frame_path = os.path.join(seq_path, frame_name)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError(f"Failed to read image: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            # Stack frames and convert to tensor
            lr_frames = np.stack(frames, axis=0)  # (3, H, W, 3)
            lr_frames = np.transpose(lr_frames, (3, 0, 1, 2))  # (3, 3, H, W)
            
            # Center frame is the target
            hr_frame = frames[1]
            hr_frame = np.transpose(hr_frame, (2, 0, 1))  # (3, H, W)
            
            return torch.from_numpy(lr_frames), torch.from_numpy(hr_frame)
            
        except Exception as e:
            # If there's an error with this sequence, try another one
            print(f"Error loading sequence {seq_path}: {e}")
            if len(self.sequences) > 1:
                return self.__getitem__(random.randint(0, len(self.sequences) - 1))
            else:
                # Create dummy data if we can't load any sequence
                dummy_lr = torch.zeros(3, 3, 256, 256)
                dummy_hr = torch.zeros(3, 256, 256)
                return dummy_lr, dummy_hr 