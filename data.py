#!/usr/bin/env python3

#data.py
#parse and chunk texts

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
    tokens = text.lower().split()
        # Initialize zero tensor of size V
    tensor = torch.zeros(len(vocab), dtype=torch.float32)
        # Fill 1s for present tokens
    for token in tokens:
        if token in vocab:
            index = vocab[token]
            tensor[index] = 1.0
            
    return tensor

#for debugging
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="encode from file path")
    parser.add_argument("text", help="text to encode")
    parser.add_argument("path", help="reference file path")
    args = parser.parse_args()
    
    print(encode(args.text, find_unique(args.path)))
