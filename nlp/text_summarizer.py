import nltk
import numpy as np
from newspaper import Article
from urllib.parse import urlparse
import logging
import os
import re
import trafilatura
import openai

# Sumy library imports for improved summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Custom tokenizer to avoid punkt_tab dependency
def simple_sent_tokenize(text):
    """
    A more robust sentence tokenizer that doesn't rely on punkt_tab.
    It handles various sentence endings and common abbreviations.
    """
    if not text:
        return []
        
    # Handle common abbreviations to avoid false sentence breaks
    common_abbr = ["Mr.", "Mrs.", "Dr.", "Ph.D.", "e.g.", "i.e.", "vs.", "etc.", "Jan.", "Feb.", "Mar.", 
                  "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "St.", "Ave.", 
                  "Inc.", "Corp.", "Ltd.", "U.S.", "U.K.", "a.m.", "p.m."]
    
    # Replace periods in common abbreviations with a special marker
    text_processed = text
    for abbr in common_abbr:
        text_processed = text_processed.replace(abbr, abbr.replace(".", "##PERIOD##"))
    
    # Regular expression for sentence boundaries
    # This looks for .!? followed by space or newline, with possible closing quotes/brackets
    sentence_endings = r'(?<=[.!?])(?=\s+[A-Z0-9])'
    
    # Split text by sentence endings
    sentences = re.split(sentence_endings, text_processed)
    
    # Also split by newlines for paragraph breaks
    expanded_sentences = []
    for s in sentences:
        # Split paragraphs but avoid splitting on newlines within sentences
        paragraph_parts = re.split(r'\n\s*\n', s)
        for part in paragraph_parts:
            if part.strip():
                expanded_sentences.append(part.strip())
    
    # Restore the original periods in abbreviations
    result = [s.replace("##PERIOD##", ".") for s in expanded_sentences]
    
    # Final cleanup - combine sentences that were falsely split
    final_result = []
    buffer = ""
    
    for s in result:
        # If the previous buffer doesn't end with sentence-ending punctuation,
        # it's likely part of the current sentence
        if buffer and not re.search(r'[.!?]$', buffer.strip()):
            buffer += " " + s
        else:
            if buffer:  # Add the completed buffer to results
                final_result.append(buffer.strip())
            buffer = s
    
    if buffer:  # Add the last buffer
        final_result.append(buffer.strip())
    
    return final_result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data (just stopwords, avoiding punkt for now)
try:
    nltk.download('stopwords', quiet=True)
    # Create directory for NLTK data if it doesn't exist
    import os
    os.makedirs(os.path.expanduser('~/nltk_data'), exist_ok=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

# Import stopwords after ensuring download
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.error(f"Error importing stopwords: {str(e)}")
    # Fallback basic stopwords if NLTK fails
    STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                    'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])

# Get API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def extract_content_from_url(url):
    """
    Extract the main content from a URL.
    
    Args:
        url (str): The URL to extract content from
        
    Returns:
        str: The extracted content
    """
    try:
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format. Please include http:// or https://")
            
        # Try using trafilatura first
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        
        # If trafilatura failed, try newspaper
        if not text:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            
        if not text:
            raise ValueError("Could not extract content from the provided URL.")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting content from URL: {str(e)}")
        raise

def textrank_summarize(text, num_sentences=None, retention_percentage=75):
    """
    Summarize text using TextRank algorithm, a graph-based ranking model
    for text processing that's similar to PageRank.
    
    Args:
        text (str): The text to summarize
        num_sentences (int, optional): Number of sentences to include in summary
                                      (calculated from retention_percentage if None)
        retention_percentage (float): Percentage of text to retain (default 75%)
        
    Returns:
        str: The summarized text
    """
    # Calculate number of sentences to keep if not specified
    if num_sentences is None:
        sentences = simple_sent_tokenize(text)
        num_sentences = max(3, int(len(sentences) * retention_percentage / 100))
    try:
        # Tokenize the text into sentences using our custom tokenizer
        sentences = simple_sent_tokenize(text)
        
        # If text is very short, return it as is
        if len(sentences) <= num_sentences:
            return text
            
        # Use our global stopwords set
        stop_words = STOPWORDS
        
        # Preprocessing: lowercase and remove special chars
        clean_sentences = []
        word_sets = []
        
        for sentence in sentences:
            # Lowercase and remove special characters
            clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            # Tokenize words
            words = clean_sentence.split()
            # Remove stopwords
            filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
            # Store word set for sentence
            word_sets.append(set(filtered_words))
            # Join words back into sentences
            clean_sentences.append(' '.join(filtered_words))
        
        # Create similarity matrix
        sentence_count = len(sentences)
        similarity_matrix = np.zeros((sentence_count, sentence_count))
        
        # Calculate sentence similarity using word overlap (Jaccard similarity)
        for i in range(sentence_count):
            for j in range(sentence_count):
                if i != j:  # Skip self-similarity
                    # Jaccard similarity: intersection / union
                    set_i = word_sets[i]
                    set_j = word_sets[j]
                    
                    if not set_i or not set_j:
                        similarity_matrix[i][j] = 0
                    else:
                        intersection = len(set_i.intersection(set_j))
                        union = len(set_i.union(set_j))
                        similarity_matrix[i][j] = intersection / union if union > 0 else 0
        
        # TextRank algorithm: Initialize sentence scores
        sentence_scores = np.ones(sentence_count)
        damping = 0.85  # Damping factor (standard for PageRank)
        threshold = 0.0001  # Convergence threshold
        iterations = 50  # Maximum iterations
        
        # Power iteration to calculate sentence scores
        prev_scores = np.zeros(sentence_count)
        for _ in range(iterations):
            for i in range(sentence_count):
                score_sum = 0
                for j in range(sentence_count):
                    if i != j and similarity_matrix[j][i] > 0:
                        # Add the score of j weighted by the similarity of j and i
                        score_sum += similarity_matrix[j][i] * sentence_scores[j]
                        
                # Update score with damping factor
                sentence_scores[i] = (1 - damping) + damping * score_sum
            
            # Check for convergence
            if np.sum(np.abs(sentence_scores - prev_scores)) < threshold:
                break
                
            prev_scores = sentence_scores.copy()
            
        # Create list of (index, score) tuples
        ranked_sentences = [(i, score) for i, score in enumerate(sentence_scores)]
        
        # Get top n sentences with highest scores
        ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = ranked_sentences[:num_sentences]
        
        # Sort by original position to maintain flow
        top_sentences.sort(key=lambda x: x[0])
        
        # Construct the summary using original sentences
        summary = ' '.join([sentences[idx] for idx, _ in top_sentences])
        
        return summary
    except Exception as e:
        logger.error(f"Error in TextRank summarization: {str(e)}")
        # Fallback to basic summarization
        try:
            # Pick first sentence, one from middle, and last sentence
            if len(sentences) >= 3:
                middle_idx = len(sentences) // 2
                summary = sentences[0] + ' ' + sentences[middle_idx] + ' ' + sentences[-1]
                return summary
            else:
                return ' '.join(sentences[:num_sentences])
        except Exception as inner_e:
            logger.error(f"Basic summarization also failed: {str(inner_e)}")
            # If all else fails, just return a portion of the text
            words = text.split()
            return ' '.join(words[:min(100, len(words))])

# Rename the function for backward compatibility
nltk_summarize = textrank_summarize

def gpt_summarize(text, max_tokens=None, retention_percentage=75):
    """
    Summarize text using OpenAI's GPT model.
    
    Args:
        text (str): The text to summarize
        max_tokens (int, optional): Maximum tokens for the summary, 
                                   calculated based on text length if None
        retention_percentage (int): What percentage of the content to retain (default 75%)
        
    Returns:
        str: The summarized text
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found, falling back to TextRank summarization")
        # Calculate number of sentences based on requested retention
        sentences = simple_sent_tokenize(text)
        target_sentences = max(3, int(len(sentences) * retention_percentage / 100))
        return textrank_summarize(text, num_sentences=target_sentences)
        
    # Truncate very long texts to fit within API limits
    max_input_chars = 12000  # OpenAI has token limits, this is a safe number of chars
    truncated_text = text[:max_input_chars] if len(text) > max_input_chars else text
    
    # Calculate target token count based on retention percentage
    word_count = len(text.split())
    
    # Average English word is about 1.3 tokens
    estimated_tokens = int(word_count * 1.3)
    
    # Set max tokens based on retention percentage (with reasonable bounds)
    if max_tokens is None:
        max_tokens = max(150, min(4000, int(estimated_tokens * retention_percentage / 100)))
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-4 for better results if available
            messages=[
                {"role": "system", "content": f"You are a professional summarizer. Create a summary that contains about {retention_percentage}% of the original text (measured by word count). Your summary should be approximately {retention_percentage} words for every 100 words in the original. Preserve the main points and essential details. Only remove redundant and less important content."},
                {"role": "user", "content": truncated_text}
            ],
            max_tokens=max_tokens,
            temperature=0.5,  # Lower temperature for more focused, deterministic output
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"Error in GPT summarization: {str(e)}")
        # Fall back to TextRank summarization
        sentences = simple_sent_tokenize(text)
        target_sentences = max(3, int(len(sentences) * retention_percentage / 100))
        return textrank_summarize(text, num_sentences=target_sentences)

def calculate_summary_stats(original_text, summary_text):
    """
    Calculate statistics about the summary.
    
    Args:
        original_text (str): The original text
        summary_text (str): The summarized text
        
    Returns:
        dict: Statistics about the summary
    """
    original_length = len(original_text.split())
    summary_length = len(summary_text.split())
    
    # Calculate reduction percentage
    if original_length > 0:
        reduction_percentage = ((original_length - summary_length) / original_length) * 100
    else:
        reduction_percentage = 0
    
    return {
        'original_length': original_length,
        'summary_length': summary_length,
        'reduction_percentage': round(reduction_percentage, 1)
    }

def sumy_summarize(text, retention_percentage=75, algorithm="lexrank"):
    """
    Summarize text using Sumy library's implementation of TextRank and LexRank.
    
    Args:
        text (str): The text to summarize
        retention_percentage (float): Percentage of sentences to keep
        algorithm (str): Either "lexrank" or "textrank"
        
    Returns:
        str: The summarized text
    """
    if not text:
        return ""
    
    # Split text into sentences
    sentences = simple_sent_tokenize(text)
    
    # If text is very short, return it as is
    if len(sentences) <= 3:
        return text
    
    # Calculate how many sentences to keep
    sentences_to_keep = max(3, int(len(sentences) * retention_percentage / 100))
    
    # Don't keep more sentences than we have
    sentences_to_keep = min(sentences_to_keep, len(sentences))
    
    # Create our own custom tokenizer using our simple_sent_tokenize function
    # This avoids the punkt_tab dependency issue
    class CustomTokenizer:
        def __init__(self, language):
            self.language = language
            
        def tokenize(self, text):
            return simple_sent_tokenize(text)
            
        def to_sentences(self, text):
            return simple_sent_tokenize(text)
            
        def to_words(self, text):
            # Split text into words (handling basic punctuation)
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            return words
    
    try:
        # Use our custom tokenizer instead of Sumy's
        custom_tokenizer = CustomTokenizer("english")
        
        # Create a parser with our custom tokenizer
        parser = PlaintextParser.from_string(text, custom_tokenizer)
        
        # Get language for stemmer and stop words
        language = "english"
        stemmer = Stemmer(language)
        stop_words = get_stop_words(language)
        
        # Choose the appropriate summarizer
        if algorithm.lower() == "lexrank":
            summarizer = LexRankSummarizer(stemmer)
        else:
            summarizer = TextRankSummarizer(stemmer)
            
        # Apply stop words
        summarizer.stop_words = stop_words
        
        # Generate summary
        summary_sentences = summarizer(parser.document, sentences_to_keep)
        
        # Convert to a string and return
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary
    except Exception as e:
        logger.error(f"Error in Sumy summarization: {str(e)}")
        # Fall back to simple extractive summarization
        return simple_extractive_summarize(text, retention_percentage)

def simple_extractive_summarize(text, retention_percentage=75):
    """
    A very simple extractive summarizer that just keeps a percentage of sentences.
    For 75% retention on a 20-sentence text, it keeps 15 sentences.

    Args:
        text (str): The text to summarize
        retention_percentage (float): Percentage of sentences to keep
        
    Returns:
        str: The summarized text
    """
    if not text:
        return ""
        
    # Split text into sentences
    sentences = simple_sent_tokenize(text)
    
    # If text is very short, return it as is
    if len(sentences) <= 3:
        return text
    
    # Calculate how many sentences to keep
    sentences_to_keep = max(3, int(len(sentences) * retention_percentage / 100))
    
    # Don't keep more sentences than we have (safety check)
    sentences_to_keep = min(sentences_to_keep, len(sentences))
    
    # For longer texts, we can apply a more aggressive reduction
    if len(sentences) > 20:
        # Calculate a sliding scale - the longer the text, the more we can reduce
        sentences_to_keep = max(10, int(len(sentences) * 0.7))
    
    if sentences_to_keep >= len(sentences):
        return text  # No reduction needed
    
    # Create summary by selecting sentences
    # Strategy: Keep first and last sentence, and evenly distributed sentences in between
    selected_sentences = []
    
    # Always keep the first sentence
    selected_sentences.append(sentences[0])
    
    # If we only want to keep 2 sentences, just return first and last
    if sentences_to_keep == 2:
        selected_sentences.append(sentences[-1])
        return " ".join(selected_sentences)
        
    # For sentences in the middle, select evenly distributed indices
    remaining_to_select = sentences_to_keep - 2  # -2 for first and last sentences
    
    if remaining_to_select > 0:
        middle_sentences = sentences[1:-1]
        
        if remaining_to_select >= len(middle_sentences):
            # If we want to keep all middle sentences, just add them all
            selected_sentences.extend(middle_sentences)
        else:
            # Otherwise, select sentences at evenly spaced intervals
            step = len(middle_sentences) / (remaining_to_select)
            for i in range(remaining_to_select):
                idx = min(int(i * step), len(middle_sentences) - 1)
                selected_sentences.append(middle_sentences[idx])
    
    # Always keep the last sentence
    selected_sentences.append(sentences[-1])
    
    # Return joined sentences
    return " ".join(selected_sentences)

def summarize_text(text):
    """
    Summarize provided text.
    
    Args:
        text (str): Text content to summarize
        
    Returns:
        dict: Summarization results
    """
    if not text:
        raise ValueError("No text provided for summarization")
    
    # Simple direct approach: reduce by 20-30% (keep 70-80%)
    retention_percentage = 75  # Middle of 70-80% range
    
    # Try multiple summarization methods in order of preference
    try:
        if OPENAI_API_KEY:
            # First choice: GPT model if API key is available
            summary = gpt_summarize(text, retention_percentage=retention_percentage)
        else:
            # Second choice: Try Sumy LexRank implementation
            try:
                summary = sumy_summarize(text, 
                                        retention_percentage=retention_percentage, 
                                        algorithm="lexrank")
            except Exception as e1:
                logger.error(f"LexRank summarization failed: {str(e1)}")
                
                # Third choice: Try Sumy TextRank implementation
                try:
                    summary = sumy_summarize(text, 
                                            retention_percentage=retention_percentage, 
                                            algorithm="textrank")
                except Exception as e2:
                    logger.error(f"TextRank summarization failed: {str(e2)}")
                    
                    # Fourth choice: Simple extractive summarizer as last resort
                    summary = simple_extractive_summarize(text, retention_percentage=retention_percentage)
    except Exception as e:
        logger.error(f"Primary summarization failed: {str(e)}")
        # Fall back to simple extraction as a last resort
        summary = simple_extractive_summarize(text, retention_percentage=retention_percentage)
    
    # Calculate statistics
    stats = calculate_summary_stats(text, summary)
    
    # If summary reduces too much or too little, adjust it
    # We want reduction to be between 20-30% (meaning retention of 70-80%)
    if stats['reduction_percentage'] < 20 or stats['reduction_percentage'] > 30:
        try:
            # Adjust the retention percentage
            if stats['reduction_percentage'] > 30:
                # Too much reduction - increase retention
                adjusted_retention = 80
            else:
                # Too little reduction - decrease retention
                adjusted_retention = 70
            
            # Try to regenerate with Sumy first, fall back if needed
            try:
                summary = sumy_summarize(text, 
                                        retention_percentage=adjusted_retention, 
                                        algorithm="lexrank")
            except Exception:
                # Fall back to simple extraction if Sumy fails
                summary = simple_extractive_summarize(text, retention_percentage=adjusted_retention)
            
            stats = calculate_summary_stats(text, summary)
        except Exception as e:
            logger.error(f"Summary length adjustment failed: {str(e)}")
    
    return {
        'summary': summary,
        'original_length': stats['original_length'],
        'summary_length': stats['summary_length'],
        'reduction_percentage': stats['reduction_percentage']
    }

def summarize_url(url):
    """
    Fetch content from URL and summarize it.
    
    Args:
        url (str): URL to extract content from and summarize
        
    Returns:
        dict: Summarization results
    """
    if not url:
        raise ValueError("No URL provided for summarization")
    
    # Extract content from URL
    content = extract_content_from_url(url)
    
    if not content:
        raise ValueError("Could not extract content from the provided URL")
    
    # Summarize the extracted content
    return summarize_text(content)