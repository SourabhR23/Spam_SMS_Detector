# import necessary libraries
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def process(text):
    pre_processed_texts = []
    for data in text:
        # remove the special characters
        remove_chars = re.sub('[^a-zA-Z]', ' ', data)
        # convert words into lowercase
        words = remove_chars.lower().split()
        # STOPWORDS list
        stops = set(stopwords.words("english"))
        # remove stopwords
        no_stopwords = [w for w in words if w not in stops]
        # Stemming the words
        words = [ps.stem(word) for word in no_stopwords]
        sent = " ".join(words)
        pre_processed_texts.append(sent.strip())

    return pre_processed_texts
