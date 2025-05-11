import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class VimeoDataset(Dataset):
    def __init__(self, root_dir, split_list=None):
        self.root_dir = os.path.join(root_dir, 'sequence')
        self.sequences = []

        if split_list is not None:
            with open(split_list, 'r') as f:
                for line in f:
                    seq = line.strip()
                    self.sequences.append(os.path.join(self.root_dir, seq))
        else:
            # fallback: scan all subfolders
            for seq_dir in os.listdir(self.root_dir):
                seq_path = os.path.join(self.root_dir, seq_dir)
                if os.path.isdir(seq_path):
                    for subseq_dir in os.listdir(seq_path):
                        subseq_path = os.path.join(seq_path, subseq_dir)
                        if os.path.isdir(subseq_path):
                            self.sequences.append(subseq_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        frames = []
        for i in range(3):
            frame_path = os.path.join(seq_path, f'im{i+1}.png')
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        lr_frames = np.stack(frames, axis=0)  # (3, H, W, 3)
        lr_frames = np.transpose(lr_frames, (3, 0, 1, 2))  # (3, 3, H, W)
        hr_frame = frames[1]
        hr_frame = np.transpose(hr_frame, (2, 0, 1))  # (3, H, W)
        return torch.from_numpy(lr_frames), torch.from_numpy(hr_frame) 