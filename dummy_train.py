import torch
from models.image_caption_model import ImageCaptionModel
from models.encoder import Encoder
from models.decoder import Decoder
import torch.nn as nn

images = torch.randn(10, 3, 224, 224)
captions = torch.randint(0, 1000, (10, 10))

print(images.shape)
print(captions.shape)

# print(images)
# print(captions)

model = ImageCaptionModel(
    embed_size=256,
    hidden_size=512,
    vocab_size=1000,
    num_layers=2,
)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD token (index 0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

batch_size = 2
for epoch in range(10):
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_captions = captions[i : i + batch_size]
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch captions shape: {batch_captions.shape}")

        # Forward pass
        outputs = model(batch_images, batch_captions)
        print(f"Model outputs shape: {outputs.shape}")

        # Prepare targets for teacher forcing
        # Remove the START token (first token) from captions
        targets = batch_captions[:, 1:]  # (batch_size, seq_length - 1)
        print(f"Targets shape: {targets.shape}")

        # Reshape outputs and targets for CrossEntropyLoss
        # outputs: (batch_size, seq_length, vocab_size) -> (batch_size * seq_length, vocab_size)
        # targets: (batch_size, seq_length) -> (batch_size * seq_length)
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
