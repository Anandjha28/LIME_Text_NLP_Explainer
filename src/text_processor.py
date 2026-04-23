import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict, List

class TextProcessor:
    def __init__(self, language: str = 'english'):
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()
    
    def get_text_statistics(self, text: str) -> Dict:
        words = text.split()
        return {
            'word_count': len(words),
            'original_length': len(text),
            'unique_words': len(set(words)),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
        }