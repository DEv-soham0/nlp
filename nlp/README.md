# Text Summarizer

A sophisticated text summarization web application leveraging Natural Language Processing (NLP) to provide concise summaries of articles and text content.

## Features

- **Dual Input Methods**: Summarize content directly from text input or via URL
- **Smart Reduction**: Maintains 70-80% of original content, preserving key information
- **Multiple Algorithms**: Uses LexRank, TextRank, and custom extractive summarization
- **Fallback System**: Cascades through multiple methods to ensure proper summary length
- **History Tracking**: Saves previously summarized content for easy reference
- **Simple UI**: Clean, intuitive interface built with Bootstrap

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **SQLite**: Database for storing summary history
- **Sumy**: For LexRank and TextRank summarization algorithms
- **Newspaper3k & Trafilatura**: For URL content extraction
- **Bootstrap**: Frontend styling

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/text-summarizer.git
   cd text-summarizer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

4. Access the application at: http://localhost:5000

## Usage

1. Select either the "Text" or "URL" tab based on your content source
2. Enter or paste your content
3. Click "Summarize"
4. View the generated summary with statistics on the reduction amount
5. Access your summary history from the History page

## How It Works

The Text Summarizer uses extractive summarization techniques to identify and keep the most important sentences from the original text. The application:

1. Tokenizes the input text into sentences
2. Applies summarization algorithms to rank sentences by importance
3. Selects the top-ranked sentences based on the target retention percentage (70-80%)
4. Reconstructs the summary while preserving sentence order
5. Adjusts summarization parameters if the reduction is outside the 20-30% target range

## License

MIT License