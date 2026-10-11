import re
import nltk
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from app.utils.logger import logger

# Ensure necessary NLTK resources are available silently
def ensure_nltk_resources():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet')
    ]
    for res_path, res_name in resources:
        try:
            nltk.data.find(res_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {res_name}")
            nltk.download(res_name, quiet=True)

ensure_nltk_resources()

class TextPreprocessor:
    def __init__(self, use_lemmatization: bool = True):
        self.use_lemmatization = use_lemmatization
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Load NLTK stopwords but preserve sentiment-bearing negation words
        raw_stopwords = set(stopwords.words('english'))
        negation_words = {
            'no', 'not', 'nor', 'neither', 'never', 'none',
            'cannot', "can't", "don't", "doesn't", "didn't",
            "won't", "wouldn't", "couldn't", "shouldn't",
            "isn't", "aren't", "wasn't", "weren't", "hasn't",
            "haven't", "hadn't", "mustn't", "needn't", "but", "very"
        }
        self.stop_words = raw_stopwords - negation_words

    def clean_text(self, text: str) -> str:
        """
        Preprocess text string:
        1. Lowercase
        2. Remove URLs
        3. Remove user mentions (@username)
        4. Clean hashtags (#hashtag -> hashtag)
        5. Remove special characters/digits but preserve letters and basic spaces
        6. Normalize whitespace
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 3. Remove mentions
        text = re.sub(r'@\w+', '', text)

        # 4. Handle hashtags (convert #word to word)
        text = re.sub(r'#(\w+)', r'\1', text)

        # 5. Remove special characters, digits, extra punct (keep single space and letters)
        text = re.sub(r'[^a-z\s]', ' ', text)

        # 6. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_filter(self, text: str) -> str:
        """
        Tokenize, remove stopwords, apply optional lemmatization, and return clean string.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""

        tokens = cleaned.split()  # Fast space-split on cleaned text

        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 and token not in self.stop_words:
                if self.use_lemmatization and self.lemmatizer:
                    token = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def transform_series(self, series) -> List[str]:
        """
        Transform a pandas Series or list of texts.
        """
        return [self.tokenize_and_filter(t) for t in series]

# Singleton instance
default_preprocessor = TextPreprocessor()
