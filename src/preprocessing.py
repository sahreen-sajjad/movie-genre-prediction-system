from functools import lru_cache
from nltk.stem import SnowballStemmer

@lru_cache(maxsize=1)
def resources():
    import spacy
    from nltk.corpus import stopwords
    return spacy.load('en_core_web_sm'), SnowballStemmer('english'), set(stopwords.words('english'))

def preprocess(text):
    """Apply the original spaCy, Snowball, lemma and stopword sequence."""
    nlp, stemmer, stop_words = resources()
    tokens = [stemmer.stem(token.text) for token in nlp(text)]
    lemmas = [doc[0].lemma_ for doc in nlp.pipe(tokens) if len(doc)]
    return ' '.join(token for token in lemmas if token not in stop_words)
