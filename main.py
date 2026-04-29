#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader

# Import your modularized code!
# (Assuming your files are named data.py, network.py, and train.py)
from data import chunk, encode
from network import LanguageDataset, LanguageClassifier
from train import train_model, evaluate_confidence
import json
from execute import build_global_vocab

if __name__ == "__main__":
    # BASE_DIR = "./files"
    # LANGUAGES = ["english", "spanish", "french"]
    # LABEL_MAP = {"english": 0, "spanish": 1, "french": 2}
    # CHUNK_SIZE = 20

    # print(">>> 1. Building Global Vocabulary...")
    # vocab = build_global_vocab(BASE_DIR, LANGUAGES, ["bible", "quote"])
    # vocab_size = len(vocab)
    # print(f"Global Vocabulary Size: {vocab_size} unique words.")

    # # ✅ Save vocab to disk so it can be reloaded later
    # with open("global_vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab, f, ensure_ascii=False, indent=2)
    # print("Vocab saved to global_vocab.json")

    # 1. Load your vocab from the JSON file
    with open("global_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # 2. Create the empty network structure
    bibleModel = LanguageClassifier(vocab_size=len(vocab), num_classes=3)
    quoteModel = LanguageClassifier(vocab_size=len(vocab), num_classes=3)

    # 3. Load the knowledge from a specific epoch (e.g., epoch 10)
    bibleModel.load_state_dict(torch.load("saved_models/bible_epoch_10.pth"))
    quoteModel.load_state_dict(torch.load("saved_models/quote_epoch_100.pth"))

    # 4. Set to evaluation mode and test!
    text = input("Enter text: ").lower().strip()
    bibleModel.eval()
    quoteModel.eval()
    evaluate_confidence("Loaded Bible Model", bibleModel, text, vocab, ["English", "Spanish", "French"])
    evaluate_confidence("Loaded Quote Model", quoteModel, text, vocab, ["English", "Spanish", "French"])