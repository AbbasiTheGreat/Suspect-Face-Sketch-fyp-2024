import main
import generator
import  dataloader
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os
import glob
import numpy as np
class Discriminator(nn.Module):

    def _init_(self, text_embedding_dim, img_shape=(3, 128, 128)):
        super(Discriminator, self)._init_()

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Calculate the input size for the linear layer
        conv_img_size = img_shape[1] // (2 ** 4)  # 4 max-pooling layers with stride 2
        self.fc_size = conv_img_size * conv_img_size * 1024

        self.fc = nn.Sequential(
            nn.Linear(self.fc_size + text_embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, img, text_embedding):
        img_features = self.model(img)
        img_features = img_features.view(img.size(0), -1)

        # Flatten or reshape text_embedding to match the number of features
        text_embedding_flat = text_embedding.view(text_embedding.size(0), -1)

        x = torch.cat((img_features, text_embedding_flat), dim=1)
        validity = self.fc(x)
        return validity