#!/usr/bin/env python3

#parse.py
#parse and chunk texts

import itertools
def chunk(ifp: str, chunk_size: int) -> list[str]:
    chunks = []
    try:
        with open(ifp, "r") as file:
            content = file.read()
            words = content.split()
            for i in range(0, len(words), chunk_size):
                current_chunk = words[i : i + chunk_size]
                chunks.append(" ".join(current_chunk))
            return chunks
    except Exception as e:
        print(f"Exception : {e}")
        return []

def find_unique(ifp: str) -> dict[str, int]:
    table = {}
    try:
        with open(ifp, "r") as file:
            content = file.readline()
            words = content.split(' ')
            i = 0
            for word in words:
                if word in table:
                    continue
                else: 
                    table[word] = i
                    i+=1
            return table
    except Exception as e:
        print(f"Exception : {e}")
        return []


from typing import Dict, List
import torch

def encode(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    """
    Converts a text chunk into a binary PyTorch tensor based on vocabulary presence.
    
    Args:
        text: The input string.
        vocab: Dictionary mapping words (str) to indices (int).
        
    Returns:
        A torch.Tensor of size (len(vocab),) with 1s for present words, 0s otherwise.
    """
    # 1. Tokenization (basic lowercase, space-split)
    tokens = text.lower().split()
    
    # 2. Initialize zero tensor of size V
    tensor = torch.zeros(len(vocab), dtype=torch.float32)
    
    # 3. Fill 1s for present tokens
    for token in tokens:
        if token in vocab:
            index = vocab[token]
            tensor[index] = 1.0
            
    return tensor

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="encode from file path")
    parser.add_argument("text", help="text to encode")
    parser.add_argument("path", help="reference file path")
    args = parser.parse_args()
    
    print(encode(args.text, find_unique(args.path)))
