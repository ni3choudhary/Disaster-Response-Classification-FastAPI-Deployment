# Importing Required Libaries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
	nltk.download('stopwords')
	nltk.download('punkt')
	nltk.download('wordnet')
	nltk.download('omw-1.4')
except FileExistsError:
    print('NLTK Data Already Exists.')
    
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

