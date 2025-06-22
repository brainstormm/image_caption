import torch
import re
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.word_freq = Counter()

    def build_vocab(self, captions):
        """Build vocabulary from captions"""
        for caption in captions:
            words = caption.lower().split()
            self.word_freq.update(words)

        # Add words that appear at least min_freq times
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, caption, max_length=None):
        """Convert caption text to token indices"""
        words = caption.lower().split()
        tokens = ["<START>"] + words + ["<END>"]

        # Convert to indices
        indices = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]

        # Pad or truncate to max_length
        if max_length:
            if len(indices) < max_length:
                indices += [self.word2idx["<PAD>"]] * (max_length - len(indices))
            else:
                indices = indices[:max_length]

        return indices

    def decode(self, indices):
        """Convert token indices back to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx["<PAD>"]:
                break
            if idx in self.idx2word:
                words.append(self.idx2word[idx])
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class ImageCaptionDataset(Dataset):
    def __init__(
        self, captions_file, images_dir, vocab=None, max_length=50, transform=None
    ):
        self.images_dir = images_dir
        self.max_length = max_length
        self.transform = transform
        self.captions = []
        self.vocab = vocab
        self.load_captions(captions_file)

        # Build vocabulary if not provided
        if self.vocab is None:
            self.vocab = Vocabulary()
            caption_texts = [caption[1] for caption in self.captions]
            self.vocab.build_vocab(caption_texts)

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
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                image = image.resize((224, 224))
                image = np.array(image)
                image = image / 255.0
                image = torch.from_numpy(image).float()
                # Convert to channels-first format (C, H, W)
                image = image.permute(2, 0, 1)

            # Tokenize and pad caption
            caption_tokens = self.vocab.encode(caption_text, self.max_length)
            caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

            return image, caption_tensor, caption_text
        else:
            raise FileNotFoundError(f"Image {image_path} not found")


def collate_fn(batch):
    """Custom collate function for DataLoader to handle variable length captions"""
    images, captions, caption_texts = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Pad captions to max length in this batch
    max_length = max(len(caption) for caption in captions)
    padded_captions = []

    for caption in captions:
        if len(caption) < max_length:
            # Pad with PAD token (0)
            padded = torch.cat(
                [caption, torch.zeros(max_length - len(caption), dtype=torch.long)]
            )
        else:
            padded = caption
        padded_captions.append(padded)

    captions = torch.stack(padded_captions, dim=0)

    return images, captions, caption_texts


# Legacy code for backward compatibility
train_data = []
test_data = []
all_data = []
captions = []
bad_data = []


def load_data():
    load_captions()
    load_images(captions)


def load_captions():
    with open("data/captions.txt", "r") as f:
        for line in f:
            caption = re.split(r"(.+\.jpg)", line.strip())
            caption = [caption[1], caption[2]] if len(caption) > 2 else caption
            if len(caption) == 2:
                captions.append(caption)
            else:
                print(caption)
                bad_data.append(caption)
    return captions


def load_images(captions):
    for caption in captions:
        image_name = caption[0]
        image_path = os.path.join("data/images", image_name)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image = np.array(image)
            image = image / 255.0
            image = torch.from_numpy(image).float()
            train_data.append(image)
            all_data.append(image)
        else:
            print(f"Image {image_path} not found")
            bad_data.append(image_name)


# Test the new dataset approach
if __name__ == "__main__":
    # Create dataset with vocabulary
    dataset = ImageCaptionDataset("data/captions.txt", "data/images", max_length=50)
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.vocab)}")

    # Test loading a single sample
    if len(dataset) > 0:
        image, caption_tokens, caption_text = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Caption text: {caption_text}")
        print(f"Caption tokens: {caption_tokens}")
        print(f"Token length: {len(caption_tokens)}")

        # Test decoding
        decoded = dataset.vocab.decode(caption_tokens.tolist())
        print(f"Decoded caption: {decoded}")

    # Test with DataLoader and collate_fn
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for batch_idx, (images, captions, texts) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"Sample captions: {texts[:2]}")
        break  # Just show first batch
