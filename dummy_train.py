import torch
from models.image_caption_model import ImageCaptionModel
from models.encoder import Encoder
from models.decoder import Decoder
import torch.nn as nn
from config import (
    embed_size,
    hidden_size,
    vocab_size,
    num_layers,
    batch_size,
    epochs,
    learning_rate,
    device,
)

images = torch.randn(100, 3, 224, 224)
captions = torch.randint(0, 1000, (100, 10))

print(images.shape)
print(captions.shape)

# print(images)
# print(captions)

model = ImageCaptionModel(
    embed_size=embed_size,
    hidden_size=hidden_size,
    vocab_size=vocab_size,
    num_layers=num_layers,
)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore START token (index 0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_captions = captions[i : i + batch_size]
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch captions shape: {batch_captions.shape}")

        # Forward pass
        outputs = model(batch_images, batch_captions)
        print(f"Model outputs shape: {outputs.shape}")

        # Remove the START token (first token) from captions
        targets = batch_captions[:, 1:]  # (batch_size, max_caption_length - 1)
        print(f"Targets shape: {targets.shape}")

        # outputs: (batch_size, max_caption_length -1, vocab_size) -> (batch_size * (max_caption_length -1), vocab_size)
        # targets: (batch_size, max_caption_length - 1) -> (batch_size * (max_caption_length - 1))
        batch_size, seq_length, vocab_size = outputs.shape
        outputs = outputs.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        print(f"Reshaped outputs shape: {outputs.shape}")
        print(f"Reshaped targets shape: {targets.shape}")

        # Calculate loss
        loss = criterion(outputs, targets)
        print(f"Loss: {loss.item()}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
    break
