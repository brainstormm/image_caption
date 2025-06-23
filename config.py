import torch

seed = 1337
torch.manual_seed(seed)

# Encoder Config.
embed_size = 256

# Decoder Config.
num_layers = 1
batch_size = 32
epochs = 1000
hidden_size = 128
learning_rate = 0.001
input_size = 256  # Same as embed_size because the image will be superimposed on the embedding of each of the words.
vocab_size = 1000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
