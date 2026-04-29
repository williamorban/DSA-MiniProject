#!/usr/bin/env python3
import os
from data import chunk, encode

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

def load_and_label_data(base_dir, file_type, languages, label_map, chunk_size=20):
    """Loads chunks for a specific file type (e.g., 'quote') across all languages."""
    all_chunks = []
    all_labels = []
    
    for lang in languages:
        path = os.path.join(base_dir, lang, f"{file_type}.txt")
        if os.path.exists(path):
            lang_chunks = chunk(path, chunk_size)
            
            all_chunks.extend(lang_chunks)
            
            # Add the corresponding label (e.g., 0, 1, or 2) for each chunk added
            label_id = label_map[lang]
            all_labels.extend([label_id] * len(lang_chunks))
            
    return all_chunks, all_labels