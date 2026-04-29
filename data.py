#!/usr/bin/env python3

#parse.py
#parse and chunk texts

import itertools
def chunk(ifp: str, chunk_size: int) -> list[str]:
    chunks = []
    try:
        with open(ifp, "r") as file:
            content = file.readline()
            words = content.split(' ')
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


def encode(table: dict[str, int]) -> None:
    pass
