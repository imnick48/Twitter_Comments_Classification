import string
from nltk.stem import PorterStemmer
import pickle
st=PorterStemmer()
def preprocess(text):
    x=" ".join([st.stem(x) for x in text.split()])
    x=x.translate(str.maketrans('','',string.punctuation))
    return x


