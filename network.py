#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import data

class LanguageDataset(Dataset):
    def __init__(self, data_chunks, labels, vocab):

        self.data_chunks = data_chunks #text chunks
        self.labels = labels #language label i.e. english spanish french (int)
        self.vocab = vocab #hash table of encoded vocabulary

    def __len__(self):
        return len(self.data_chunks)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        # Get the text chunk and its corresponding label
        chunk = self.data_chunks[idx]
        label = self.labels[idx]
        
        tensor = data.encode(chunk, self.vocab)
        
        # PyTorch requires labels to be long tensors
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return tensor, label_tensor
    
#neural network design:
class LanguageClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=3):
        super(LanguageClassifier, self).__init__()

        self.layer1 = nn.Linear(vocab_size, 128) #hidden layer
        
        # Activation function to introduce non-linearity (helps the network learn complex patterns)
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(128, num_classes) #output layer, outputs 3 numbers

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        logits = self.output_layer(x) # Raw, unnormalized scores
        return logits