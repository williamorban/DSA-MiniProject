#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader

# Import your modularized code!
# (Assuming your files are named data.py, network.py, and train.py)
from data import chunk, encode
from network import LanguageDataset, LanguageClassifier
from train import train_model, evaluate_confidence

# ==========================================
# Helper: Build One Master Dictionary
# ==========================================
def build_global_vocab(base_dir, languages, file_types):
    """Scans every file in the dataset to create one unified Hash Map."""
    global_vocab = {}
    current_index = 0
    
    for lang in languages:
        for ftype in file_types:
            path = os.path.join(base_dir, lang, f"{ftype}.txt")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        words = file.read().split()
                        for word in words:
                            if word not in global_vocab:
                                global_vocab[word] = current_index
                                current_index += 1
                except Exception as e:
                    print(f"Error reading {path}: {e}")
            else:
                print(f"Warning: File not found -> {path}")
                
    return global_vocab

# ==========================================
# Helper: Load and Label Chunks
# ==========================================
def load_and_label_data(base_dir, file_type, languages, label_map, chunk_size=20):
    """Loads chunks for a specific file type (e.g., 'quote') across all languages."""
    all_chunks = []
    all_labels = []
    
    for lang in languages:
        path = os.path.join(base_dir, lang, f"{file_type}.txt")
        if os.path.exists(path):
            # Call your chunking function from data.py
            lang_chunks = chunk(path, chunk_size)
            
            # Add these chunks to our master list
            all_chunks.extend(lang_chunks)
            
            # Add the corresponding label (e.g., 0, 1, or 2) for EVERY chunk we just added
            label_id = label_map[lang]
            all_labels.extend([label_id] * len(lang_chunks))
            
    return all_chunks, all_labels

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Configuration
    BASE_DIR = "./files"
    LANGUAGES = ["english", "spanish", "french"]
    LABEL_MAP = {"english": 0, "spanish": 1, "french": 2}
    CHUNK_SIZE = 20 # Words per chunk

    print(">>> 1. Building Global Vocabulary...")
    # This reads all bibles and all quotes into one dictionary
    vocab = build_global_vocab(BASE_DIR, LANGUAGES, ["bible", "quote"])
    vocab_size = len(vocab)
    print(f"Global Vocabulary Size: {vocab_size} unique words.")

    print("\n>>> 2. Chunking Data...")
    # Load Bible data (Classic)
    bible_chunks, bible_labels = load_and_label_data(BASE_DIR, "bible", LANGUAGES, LABEL_MAP, CHUNK_SIZE)
    # Load Quote data (Modern)
    quote_chunks, quote_labels = load_and_label_data(BASE_DIR, "quote", LANGUAGES, LABEL_MAP, CHUNK_SIZE)
    
    print(f"Loaded {len(bible_chunks)} Bible chunks.")
    print(f"Loaded {len(quote_chunks)} Quote chunks.")

    print("\n>>> 3. Setting up PyTorch Datasets & DataLoaders...")
    bible_dataset = LanguageDataset(bible_chunks, bible_labels, vocab)
    quote_dataset = LanguageDataset(quote_chunks, quote_labels, vocab)
    
    # Batch size dictates how many chunks the AI reviews before updating its weights
    bible_loader = DataLoader(bible_dataset, batch_size=32, shuffle=True)
    quote_loader = DataLoader(quote_dataset, batch_size=32, shuffle=True)

    print("\n>>> 4. Initializing Neural Networks...")
    model_classic = LanguageClassifier(vocab_size=vocab_size, num_classes=3)
    model_modern = LanguageClassifier(vocab_size=vocab_size, num_classes=3)

    print("\n>>> 5. Training Models...")
    print("--- Training Bible/Classic Model ---")
    train_model(model_classic, bible_loader, epochs=10, save_prefix="bible") # Start with 10 epochs to test speed
    
    print("\n--- Training Quote/Modern Model ---")
    train_model(model_modern, quote_loader, epochs=100, save_prefix="quote")

    print("\n>>> 6. THE CONFIDENCE SHOWDOWN")
    # A test sentence combining archaic phrasing with modern concepts
    test_sentence = "thou shalt look before you leap" 
    print(f"Test Sentence: '{test_sentence}'")
    
    # We pass 'vocab' so evaluate_confidence uses the same encoding map!
    evaluate_confidence("Model A (Trained on Bibles)", model_classic, test_sentence, vocab, ["English", "Spanish", "French"])
    evaluate_confidence("Model B (Trained on Quotes)", model_modern, test_sentence, vocab, ["English", "Spanish", "French"])