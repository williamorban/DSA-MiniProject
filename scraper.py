#!/usr/bin/env python3

#scraper.py
#scrape french bible website, pdf

import os
import io
import re
from pypdf import PdfReader
import pdfplumber


def read_pdf(path: str, out: str):
    skip_pages = [i for i in range(35)]
    try:
        text_content = []
        with pdfplumber.open(path) as pdf, open(out, "a") as outF:
            for i, page in enumerate(pdf.pages):
                # Define core area: ignore top 50 points and bottom 50 points
                # Coordinates: (left, top, right, bottom)
                core_bbox = (0, 50, page.width, page.height - 50)
                
                # Crop and extract text
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
    
def clean_text(text: str) -> str:
    # return re.sub(r'[^\D\n]', ' ', text).strip()
    return re.sub(r'[\d\W_]+', ' ', text).strip().lower()

    

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape a pdf.")
    parser.add_argument("path", help="pdf path")
    parser.add_argument("out", help="out path")
    args = parser.parse_args()
    
    read_pdf(args.path, args.out)
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