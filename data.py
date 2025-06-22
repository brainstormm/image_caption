import torch
import re
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    def __init__(self, captions_file, images_dir):
        self.images_dir = images_dir
        self.captions = []
        self.load_captions(captions_file)

    def load_captions(self, captions_file):
        with open(captions_file, "r") as f:
            for line in f:
                caption = re.split(r"(.+\.jpg)", line.strip())
                caption = [caption[1], caption[2]] if len(caption) > 2 else caption
                if len(caption) == 2:
                    self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name, caption_text = self.captions[idx]
        image_path = os.path.join(self.images_dir, image_name)

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image)
            image = image / 255.0
            image = torch.from_numpy(image).float()
            # Convert to channels-first format (C, H, W)
            image = image.permute(2, 0, 1)

            return image, caption_text
        else:
            raise FileNotFoundError(f"Image {image_path} not found")
