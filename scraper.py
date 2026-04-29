#!/usr/bin/env python3

#scraper.py
#scrape and format pdfs/txt files

import os
import io
import re
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse, urldefrag
import pdfplumber
from collections import deque

# for spanish
"""
def read_pdf(path: str, out: str):
    # skip_pages = [i for i in range(35)]
    skip_pages = [i for i in range(10)]
    stop_page = 1324
    try:
        text_content = []
        with pdfplumber.open(path) as pdf, open(out, "a") as outF:
            for i, page in enumerate(pdf.pages):
                # Define core area: ignore top 50 points and bottom 50 points
                # Coordinates: (left, top, right, bottom)
                core_bbox = (0, 50, page.width, page.height - 50)
                
                # Crop and extract text
                if i >= stop_page: 
                    print("stop found, break")
                    break
                if i not in skip_pages:
                    print(f"Extracting page {i+1}...")

                    cropped_page = page.within_bbox(core_bbox)
                    text = cropped_page.extract_text()
                    if text:
                        text = clean_text(text)
                        outF.write(text)
        
    except Exception as e:
        print(f"Error fetching PDF : {e}")
        return None
    """

    
# def clean_text(text: str) -> str:
#     # return re.sub(r'[^\D\n]', ' ', text).strip()
#     # return re.sub(r'[\d\W_]+', ' ', text).strip().lower()
#     return re.sub(r'[a-zA-ZÀ-ÿœŒ]+(?:[' -][a-zA-ZÀ-ÿœŒ]+)*', ' ', text).strip().lower()

#for french
def read_pdf(path: str, out: str):
    """
    Reads a two-column PDF page by page, splitting each page in half vertically.
    Extracts the left column, then the right column, and cleans the text.
    """
    skip_pages = [i for i in range(10)]
    stop_page = 1333
    try:
        with pdfplumber.open(path) as pdf, open(out, "a", encoding="utf-8") as outF:
            for i, page in enumerate(pdf.pages):
                if i >= stop_page: 
                    print("stop found, break")
                    break
                if i not in skip_pages:
                    print(f"Extracting page {i+1}...")

                    # 1. Define bounds for the left and right columns
                    width = page.width
                    height = page.height
                    
                    # Coordinates: (left, top, right, bottom)
                    # Splitting the width in half (width / 2) creates our column boundary
                    left_bbox = (0, 50, width / 2, height - 10)
                    right_bbox = (width / 2, 50, width, height - 10)
                    
                    # 2. Crop and extract text from the left column
                    left_crop = page.within_bbox(left_bbox)
                    left_text = left_crop.extract_text() or ""
                    
                    # 3. Crop and extract text from the right column
                    right_crop = page.within_bbox(right_bbox)
                    right_text = right_crop.extract_text() or ""
                    
                    # 4. Combine text: left column followed by right column
                    # A newline connects them so `clean_text` can fix hyphenation 
                    # bridging the bottom of the left column to the top of the right
                    full_text = left_text + "\n" + right_text
                    
                    if full_text.strip():
                        # Pass through our previously built cleaner!
                        text = clean_text(full_text)
                        outF.write(text + " ")
        
    except Exception as e:
        print(f"Error fetching PDF : {e}")
        return None

def clean_text(text: str) -> str:
    """
    Cleans French text by handling page break hyphens, keeping 
    accents, and preserving elisions/compound words.
    """
    # 1. Join words that are split across lines by a hyphen
    # Example: "miséri-\ncorde" becomes "miséricorde"
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 2. Extract valid words using a pattern that allows internal hyphens and apostrophes.
    # - [a-zA-ZÀ-ÿœŒ]+ : Starts with one or more letters (including French accents/ligatures)
    # - (?:['’\-][a-zA-ZÀ-ÿœŒ]+)* : Optionally followed by an apostrophe or hyphen and more letters
    pattern = r"[a-zA-ZÀ-ÿœŒ]+(?:['’\-][a-zA-ZÀ-ÿœŒ]+)*"
    words = re.findall(pattern, text)
    
    # 3. Join back into a single string with spaces and convert to lowercase
    return ' '.join(words).lower()


import argparse
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Scrape a pdf.")
    parser.add_argument("path", help="pdf path")
    parser.add_argument("out", help="out path")
    args = parser.parse_args()
    
    read_pdf(args.path, args.out)

#for english conversion (use spanish clean fx)
"""
    parser = argparse.ArgumentParser(description="Clean a txt file")
    parser.add_argument("path", help="txt path")
    args = parser.parse_args()
    
    with open(args.path, "r+") as file:
        lines = file.readlines()
        lines = clean_text(" ".join(lines))
        file.seek(0)
        file.write(lines)
        file.truncate()
"""