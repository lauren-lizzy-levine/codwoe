##lemmatization, stemming, lowercase, and dropping stop words and punctuations
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
def rem_punc(sent):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    nopunc = []
    for w in sent:
        if w not in punc:
            nopunc.append(w)
    return nopunc

def rem_stop(sent):
    stop_words = set(stopwords.words('english'))
    nostop = []
    for w in sent:
        if w.lower() not in stop_words:
            nostop.append(w)
    return nostop

def lemmatize(sent):
    lemmatizer = WordNetLemmatizer()
    lemmed = []
    for w in sent:
        lemmed.append(lemmatizer.lemmatize(w))
    return lemmed

def stem(sent):
    ps = PorterStemmer()
    stemmed = []
    for w in sent:
        stemmed.append(ps.stem(w))
    return stemmed


def lower(sent):
    lowered = []
    for w in sent:
        lowered.append(w.lower())
    return lowered


