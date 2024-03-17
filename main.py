import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import torch.nn as nn  # This import is redundant, already imported from torch
from PIL import Image
import os
import glob
from transformers import BertModel
import numpy as np
import matplotlib.pyplot as plt
import User_interface

# Generator and Discriminator (assuming they are defined elsewhere)
from generator import Generator
from discriminator import Discriminator

# Assuming dataloader is defined and loaded as shown in previous parts
from dataloader import ImageTextDataset  # Import only if dataloader is used

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(text_embedding_dim=768, noise_dim=100).to(device)
discriminator = Discriminator(text_embedding_dim=768).to(device)

# Define the training function
def train_gan(generator, discriminator, data_loader, num_epochs):
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # ... rest of the training function

# Train the GAN (assuming dataloader is defined)
train_gan(generator, discriminator, dataloader, num_epochs=10)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Move BERT model to device
bert_model = bert_model.to(device)

# Text description and encoding
text_description = """
this woman has a square face with long hair.
 she has a pair of big normal eyes, with dense thick and flat eyebrows.
 her mouth is thin and wide, with a medium normal nose and her ears are big.
 she has glasses and hasn’t beard.

"""
encoded_text = tokenizer(text_description, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)

# Generate text embedding
with torch.no_grad():
    text_embedding = bert_model(**encoded_text).last_hidden_state[:, 0, :]

# Generate image
noise = torch.randn(1, 100).to(device)
generator.eval()

with torch.no_grad():
    generated_image = generator(noise, text_embedding)

    generated_image = generated_image.cpu().detach().numpy()
    generated_image = np.transpose(generated_image[0], (1, 2, 0))
    generated_image = (generated_image + 1) / 2
    generated_image = np.clip(generated_image, 0, 1)

# Display image
plt.imshow(generated_image)
plt.show()
