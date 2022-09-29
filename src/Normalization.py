import itertools
import unicodedata
import string
import re
import nltk 
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer

# Declare vars
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
punct_all = r"[!@#$%^&*()[]{};:,./<>?\|`~-=_+]"

class Normalization():
    """
        This Class will be used to apply text preprocessing to input text data
        :param str corpus: The corpus text data
        :returns: The answer
        :rtype: str
    """
    def __init__(self) -> None:
        pass

    def is_stop_word(self, token):
        return token.lower() in stopwords

    def handle_stopw(self, data):
        return ' '.join([i for i in word_tokenize(data.lower()) if i not in stopwords])

    def clean_text(self, text):
        text = re.sub(' +', ' ', text) # remove extra spaces btw words
        text = text.strip() # remore spaces at beggining and end
        text = text.lower() # lower case text
        return text

    def handle_contractions(self, text):
        # specific
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        # general
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text

    def part_of_speech_tag(self, text_token):
        return nltk.pos_tag(text_token)

    def lemmatization(self, word, tag):
        return lemmatizer.lemmatize(word, tag).lower()

    def stemming(self, word, tag):
        return stemmer.stem(word)

    def word_tokenize(self, text):
        return nltk.word_tokenize(text.lower())

    def baseline_prep(self, text, norm='stem'): 
        response = []
        text = str(text)
        text = self.clean_text(text)
        text = self.handle_contractions(text)
        list_text = [word for word in nltk.word_tokenize(text.lower())]
        text = nltk.pos_tag(list_text) 

        return response
