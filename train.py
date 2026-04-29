import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import os
import time

from data import encode

#training loop (AI generated)
def train_model(model, dataloader, epochs=50, save_prefix=None): #default of 50 epochs
    # CrossEntropyLoss automatically applies Softmax internally during training!
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer acts as the "learner" updating the weights
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # saved models
    if save_prefix and not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    model.train() # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.perf_counter() #timing
        for batch_tensors, batch_labels in dataloader:
            # 1. Clear old gradients
            optimizer.zero_grad()
            
            # 2. Forward pass (make a prediction)
            predictions = model(batch_tensors)
            
            # 3. Calculate how wrong the prediction was
            loss = criterion(predictions, batch_labels)
            
            # 4. Backward pass (calculate gradients)
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress every epoch
        end_time = time.perf_counter()
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Time elapsed: {end_time-start_time:.4f} seconds")

        # Save the model's knowledge (state_dict) at the end of each epoch
        if save_prefix:
            save_path = os.path.join("saved_models", f"{save_prefix}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

import matplotlib.patches as mpatches
import numpy as np
def evaluate_confidence(model_name, model, test_sentence, vocab, languages):
    confidenceList = []
    model.eval() # evaluation mode (turns off training behaviors)
    
    #input
    input_tensor = encode(test_sentence, vocab).unsqueeze(0) # unsqueeze adds a "batch" dimension
    
    #oputput logits
    with torch.no_grad(): # Don't track gradients for memory efficiency
        logits = model(input_tensor)
        
    #convert logits w/ confidence calc
    softmax = nn.Softmax(dim=1)
    confidences = softmax(logits)[0] # Extract the first (and only) item in the batch
    
    print(f"\n--- {model_name} Confidence ---")
    for i, lang in enumerate(languages):
        confidence = round(confidences[i].item() * 100.00, 2)
        confidenceList.append(confidence)
        print(f"{lang}: {confidence}%")

    # 4. Bar plot
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 4))
    

    bars = ax.bar(languages, confidenceList, color=colors[:len(languages)], width=0.5, zorder=2)

    # Label each bar with its percentage
    for bar, val in zip(bars, confidenceList):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Confidence (%)", fontsize=11)
    ax.set_title(f"{model_name}\n\"{test_sentence}\"", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_')}_confidence.png", dpi=150)
    plt.show()
    
    return confidenceList