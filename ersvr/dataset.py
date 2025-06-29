import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class VimeoDataset(Dataset):
    def __init__(self, root_dir, split_list=None, sample_size=None, verbose=True):
        self.root_dir = root_dir
        self.verbose = verbose
        self.sequences = []
        
        seq_dir = None
        
        vimeo_septuplet = os.path.join(root_dir, "vimeo_settuplet_1")
        if os.path.exists(vimeo_septuplet) and os.path.isdir(vimeo_septuplet):
            if verbose:
                print(f"Found vimeo_settuplet_1 at {vimeo_septuplet}")
            
            sequences_dir = os.path.join(vimeo_septuplet, "sequences")
            if os.path.exists(sequences_dir) and os.path.isdir(sequences_dir):
                seq_dir = sequences_dir
                if verbose:
                    print(f"Found sequences directory at {seq_dir}")
            else:
                seq_dir = vimeo_septuplet
        else:
            for dirname in ["sequence", "sequences"]:
                candidate = os.path.join(self.root_dir, dirname)
                if os.path.exists(candidate) and os.path.isdir(candidate):
                    seq_dir = candidate
                    if verbose:
                        print(f"Found {dirname} directory at {seq_dir}")
                    break
        
        if seq_dir is None:
            seq_dir = self.root_dir
            if verbose:
                print(f"No specific sequence directory found, using root: {seq_dir}")
        
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
            
            content = os.listdir(seq_dir)
            subdirs = [d for d in content if os.path.isdir(os.path.join(seq_dir, d))]
            
            if verbose:
                print(f"Found {len(subdirs)} items in {seq_dir}")
                if subdirs:
                    print(f"First few items: {subdirs[:5]}")
            
            if self._check_sequence_valid(seq_dir):
                if verbose:
                    print(f"Found valid sequence at root level: {seq_dir}")
                self.sequences.append(seq_dir)
            else:
                numeric_dirs = [d for d in subdirs if d.isdigit() or (len(d) >= 5 and d[:5].isdigit())]
                
                if numeric_dirs and verbose:
                    print(f"Found {len(numeric_dirs)} numeric directories (e.g., {numeric_dirs[:5]})")
                
                if numeric_dirs:
                    for dir_name in numeric_dirs:
                        dir_path = os.path.join(seq_dir, dir_name)
                        if self._check_sequence_valid(dir_path):
                            self.sequences.append(dir_path)
                        else:
                            for subdir in os.listdir(dir_path):
                                subdir_path = os.path.join(dir_path, subdir)
                                if os.path.isdir(subdir_path) and self._check_sequence_valid(subdir_path):
                                    self.sequences.append(subdir_path)
                else:
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
        
        if self._check_sequence_valid(directory):
            self.sequences.append(directory)
            return
        
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
        patterns = [
            [f'im{i}.png' for i in range(1, 4)],
            [f'im{i:02d}.png' for i in range(1, 4)],
            [f'frame{i:03d}.png' for i in range(1, 4)],
            [f'{i:02d}.png' for i in range(1, 4)]
        ]
        
        for pattern in patterns:
            valid = True
            for frame_name in pattern:
                frame_path = os.path.join(seq_path, frame_name)
                if not os.path.isfile(frame_path):
                    valid = False
                    break
            if valid:
                self.file_pattern = pattern
                return True
                
        return False

    def __len__(self):
        return max(1, len(self.sequences))
    
    def __getitem__(self, idx):
        if len(self.sequences) == 0:
            dummy_lr = torch.zeros(3, 3, 256, 256)
            dummy_hr = torch.zeros(3, 256, 256)
            return dummy_lr, dummy_hr
        
        if idx >= len(self.sequences):
            idx = idx % len(self.sequences)
            
        seq_path = self.sequences[idx]
        frames = []
        
        try:
            if not hasattr(self, 'file_pattern'):
                self._check_sequence_valid(seq_path)
            
            if not hasattr(self, 'file_pattern'):
                self.file_pattern = [f'im{i}.png' for i in range(1, 4)]
            
            for frame_name in self.file_pattern:
                frame_path = os.path.join(seq_path, frame_name)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError(f"Failed to read image: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            lr_frames = np.stack(frames, axis=0)
            lr_frames = np.transpose(lr_frames, (3, 0, 1, 2))
            
            hr_frame = frames[1]
            hr_frame = np.transpose(hr_frame, (2, 0, 1))
            
            return torch.from_numpy(lr_frames), torch.from_numpy(hr_frame)
            
        except Exception as e:
            print(f"Error loading sequence {seq_path}: {e}")
            if len(self.sequences) > 1:
                return self.__getitem__(random.randint(0, len(self.sequences) - 1))
            else:
                dummy_lr = torch.zeros(3, 3, 256, 256)
                dummy_hr = torch.zeros(3, 256, 256)
                return dummy_lr, dummy_hr 