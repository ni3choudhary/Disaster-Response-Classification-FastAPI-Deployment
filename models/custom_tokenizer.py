# Importing Required Libaries
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# stopword to remove from data
stop_words = set(stopwords.words('english')) 

# Extract Features from tex
def tokenize(text):
    # word tokenize 
    tokens = word_tokenize(text)
    # lemmatizer to return the base word
    lemmatizer = WordNetLemmatizer()

    # list of clean tokens after removing stopwords and punctuation from tokens
    clean_tokens = [ lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stop_words and token.isalpha() ]
    
    return clean_tokens

