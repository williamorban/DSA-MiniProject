#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import data

# ==========================================
# Step 5: The PyTorch Dataset Class
# ==========================================
class LanguageDataset(Dataset):
    def __init__(self, data_chunks, labels, vocab):
        """
        Args:
            data_chunks (list of str): The list of text chunks.
            labels (list of int): The language labels (e.g., 0 for EN, 1 for ES, 2 for FR).
            vocab (dict): Your vocabulary hash map to encode the text.
        """
        self.data_chunks = data_chunks
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        """Returns the total number of samples in the dataset."""
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

# ==========================================
# Step 6: Designing the Neural Network
# ==========================================
class LanguageClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=3):
        super(LanguageClassifier, self).__init__()
        
        # Input layer: Takes in the Bag of Words tensor (size of vocab)
        # Hidden layer: Arbitrary size (e.g., 128 neurons), you can experiment with this!
        self.layer1 = nn.Linear(vocab_size, 128)
        
        # Activation function to introduce non-linearity (helps the network learn complex patterns)
        self.relu = nn.ReLU()
        
        # Output layer: Takes the 128 hidden neurons and outputs exactly 3 numbers
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        """Defines how data passes through the network."""
        x = self.layer1(x)
        x = self.relu(x)
        logits = self.output_layer(x) # Raw, unnormalized scores
        return logits

# ==========================================
# Step 7: Understanding Softmax Example
# ==========================================
def test_softmax_confidence():
    print("--- Testing Softmax Confidence ---")
    
    # Imagine our model output these raw numbers (logits) for a specific sentence:
    # Index 0: English, Index 1: Spanish, Index 2: French
    raw_outputs = torch.tensor([[1.5, -0.2, 5.3]]) 
    print(f"Raw Logits: {raw_outputs.tolist()[0]}")

    # Apply Softmax to convert to percentages (confidence) that sum to 1.0
    # dim=1 means we calculate it across the columns for this specific sample
    softmax = nn.Softmax(dim=1)
    confidences = softmax(raw_outputs)
    
    # Extract the percentages
    en_conf = confidences[0][0].item() * 100
    es_conf = confidences[0][1].item() * 100
    fr_conf = confidences[0][2].item() * 100
    
    print(f"\nConfidences:")
    print(f"English: {en_conf:.2f}%")
    print(f"Spanish: {es_conf:.2f}%")
    print(f"French:  {fr_conf:.2f}%")
    print("\nNotice how the highest raw logit (5.3) becomes a highly confident percentage!")

if __name__ == "__main__":
    # Run the file to see Softmax in action!
    test_softmax_confidence()