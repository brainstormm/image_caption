import torch
from models.image_caption_model import ImageCaptionModel
from models.encoder import Encoder
from models.decoder import Decoder
from data import ImageCaptionDataset
from config import epochs, seed
from torch.utils.data import DataLoader, random_split

dataset = ImageCaptionDataset("data/captions.txt", "data/images")

print(f"Dataset size: {len(dataset)}")
image, caption = dataset[0]
print(type(dataset))
print(f"Image shape: {image.shape}")
print(f"Caption: {caption}")

total_data_size = len(dataset)
train_data_size = int(total_data_size * 0.8)
eval_data_size = int(total_data_size * 0.1)
test_data_size = total_data_size - train_data_size - eval_data_size

train_data, eval_data, test_data = random_split(
    dataset,
    [train_data_size, eval_data_size, test_data_size],
    generator=torch.Generator().manual_seed(seed),
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Eval batches: {len(eval_loader)}")
print(f"Test batches: {len(test_loader)}")

model = ImageCaptionModel(
    embed_size=256,
    hidden_size=512,
    vocab_size=1000,
    num_layers=2,
)

for epoch in range(epochs):
    for batch in train_loader:
        # batch is a tuple of (image, caption)
        # batch[0] is the image, shape: (batch_size, 3, 224, 224)
        # batch[1] is the caption, shape: (batch_size, caption_length)
        model.train()
        image, caption = batch
        print(image.shape)
        print(len(caption))
        output = model(image, caption)
        print(output.shape)
        break
    break
