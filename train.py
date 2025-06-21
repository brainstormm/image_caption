import torch
from models.encoder import Encoder

# Test individual components
print("=== Testing Encoder ===")
encoder = Encoder(embed_size=256)

# (batch_size, 3, 224, 224)
images = torch.randn(2, 3, 224, 224)
features = encoder(images)
print(f"Encoder output shape: {features.shape}")