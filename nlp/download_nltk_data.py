"""
Script to download necessary NLTK data for the text summarizer.
This should be run once before deploying the application.
"""

import nltk

# Essential NLTK resources for our summarizer
nltk_resources = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger'
]

def download_nltk_data():
    """Download all required NLTK data packages."""
    for resource in nltk_resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

if __name__ == "__main__":
    print("Starting NLTK data download...")
    download_nltk_data()
    print("NLTK data download complete!")