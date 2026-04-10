import os
import json

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset , DataLoader


class COCODataset(Dataset):
    def __init__(self , image_dir: str , captions_file: str , image_size: int = 256) -> None:
        with open(captions_file , 'r') as f:
            coco = json.load(f)

        # Building the mapping from image id -> first caption found in annotations
        id_to_caption = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in id_to_caption:
                id_to_caption[img_id] = ann['caption']

        # Building the list of (image_path , caption) pairs
        self.samples = []
        for img_info in coco['images']:
            img_id = img_info['id']
            img_path = os.path.join(image_dir , img_info['file_name'])
            if img_id in id_to_caption and os.path.exists(img_path):
                self.samples.append((img_path,id_to_caption[img_id]))

        # Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 , 0.5 , 0.5],
                [0.5 , 0.5 , 0.5]
            )
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self , idx: int) -> Tuple[object , str]:
        ''' Returns (transformed_image_tensor , caption) '''
        path , caption = self.samples[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image) , caption


def numpy_collate(batch: List[Tuple[object , str]]) -> Tuple[np.ndarray , List[str]]:
    '''
    Collate function for DataLoader.
    Converts a list of (torch.Tensor , str) into a sinlge np array of shape (B,C,H,W) and a list of caption strings.
    '''
    images , captions = zip(*batch)
    images = np.stack(
        [img.numpy() for img in images],
        axis=0
    )
    return images , list(captions)

def build_dataloader(image_dir: str , captions_file: str , batch_size: int = 128 , num_workers: int = 2 , image_size: int = 256) -> DataLoader:
    ''' Builds a DataLoader for the COCO dataset. '''
    dataset = COCODataset(image_dir , captions_file , image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        drop_last=True,
        pin_memory=False
    )