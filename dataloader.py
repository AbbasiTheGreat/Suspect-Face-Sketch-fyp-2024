import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import os
import glob
from transformers import BertModel
import numpy as np

import matplotlib.pyplot as plt


class ImageTextDataset(Dataset):
    def _init_(self, image_dir, text_dir, transform=None, bert_model=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load BERT model only once, not in _getitem_
        if bert_model is None:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert_model = bert_model
        self.bert_model.eval()  # Set the BERT model to evaluation mode

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        # Load and transform the image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load and tokenize the text
        text_path = os.path.join(self.text_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
        with open(text_path, 'r') as file:
            text = file.read().strip()
        encoded_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        # Generate text embeddings
        with torch.no_grad():
            outputs = self.bert_model(**encoded_text)
        text_embeddings = outputs.last_hidden_state[:, 0, :]  # Get the embeddings of the [CLS] token

        return image, text_embeddings


image_folder_path =  destination_folder_path + '/images'
text_folder_path =   destination_folder_path  + '/text'
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

dataset = ImageTextDataset(image_folder_path, text_folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)