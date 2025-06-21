import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(
            input_size=embed_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            num_layers=num_layers,
        )
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(vocab_size, embed_size)

    # images.shape : (batch_size, 3, 224, 224)
    # captions.shape : (batch_size , max_caption_length)
    def forward(self, images, captions) -> torch.Tensor:
        batch_size = images.shape[0]
        max_caption_length = captions.shape[1]

        # Convert images to the features
        encoded_image_features = self.encoder(images)
        # encoded_image_features.shape : (batch_size, embed_size)

        # Initialize the hidden states for the decoder
        initial_hidden = self.init_hidden(batch_size)

        # Prepare the input sequence for the decoder
        # Remove the last token from captions (we don't need to predict the end token)
        decoder_input = captions[:, :-1]  # (batch_size, max_caption_length - 1)

        # Embed the input tokens
        embedded_input = self.word_embedding(
            decoder_input
        )  # (batch_size, max_caption_length - 1, embed_size)

        # Concatenate image features with embedded input
        # Repeat image features for each time step
        image_features_expanded = encoded_image_features.unsqueeze(1).expand(
            -1, max_caption_length - 1, -1
        )
        # (batch_size, max_caption_length - 1, embed_size)

        # Combine image features with word embeddings
        decoder_input_combined = embedded_input + image_features_expanded

        # Pass through the decoder
        decoder_output, _ = self.decoder(decoder_input_combined, initial_hidden)
        # decoder_output.shape : (batch_size, vocab_size)

        return decoder_output

    def init_hidden(self, batch_size) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
