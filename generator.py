import dataloader
import discriminator
import main
import torch.nn as nn
import torch

class Generator(nn.Module):
    def _init_(self, text_embedding_dim, noise_dim, img_shape=(3, 128, 128)):
        super(Generator, self)._init_()
        self.img_shape = img_shape

        # Calculate the size of the output of the first linear layer
        self.fc_output_size = 1024 * (img_shape[1] // 16) * (img_shape[2] // 16)  # Adjusted for the size of the image after 3 transposed conv layers
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + text_embedding_dim, self.fc_output_size),  # Adjusted the output size for convolutional layers
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.fc_output_size),
            # Reshape to a tensor with shape (batch_size, 1024, img_height//16, img_width//16)
            nn.Unflatten(1, (1024, img_shape[1] // 16, img_shape[2] // 16)),
            # Convolutional layers
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, img_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        if text_embedding.dim() == 3:
            text_embedding = text_embedding[:, 0, :]
        x = torch.cat((noise, text_embedding), dim=1)
        img = self.model(x)
        return img
