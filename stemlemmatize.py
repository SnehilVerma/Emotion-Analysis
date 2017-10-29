from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


lemmatizer=WordNetLemmatizer()
stemmer=PorterStemmer()

print(lemmatizer.lemmatize("are"))
print(stemmer.stem("wants"))