#!/bin/bash

# Download spaCy English model
python -m spacy download en_core_web_sm

# Optional: Download NLTK data if you use it
python -m nltk.downloader punkt
