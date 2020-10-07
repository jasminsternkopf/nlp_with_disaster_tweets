import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS

# Functions to preprocess the data:

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def stopwords_(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

lemmatizer = WordNetLemmatizer()
def lemmatizer_(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess(X,lowercase=True,remove_punct=True,remove_stopwords=True,use_lemmatizer=True):
    if lowercase:
        X=X.str.lower()
    if remove_punct:
        X=X.apply(lambda text: remove_punctuation(text))
    if remove_stopwords:
        X=X.apply(lambda text: stopwords_(text))
    if use_lemmatizer:
        X=X.apply(lambda text: lemmatizer_(text))
    return X

def vectorize(X_train, X_test, min_tdf): # turns the text-vectors into Tfidf-arrays
    vectorizer=TfidfVectorizer(min_df=min_tdf)
    X_train_tfidf=vectorizer.fit_transform(X_train)
    X_train_tfidf=X_train_tfidf.toarray()
    X_test_tfidf=vectorizer.transform(X_test)
    X_test_tfidf=X_test_tfidf.toarray()
    return X_train_tfidf,X_test_tfidf

def get_data(min_tdf=3): # reads the csv files and returns the preprocessed and vectorized train and test set
    df=pd.read_csv('data/train.csv')
    df_Xtest=pd.read_csv('data/test.csv')
    df_ytest=pd.read_csv('data/submission.csv')

    X_train=df.loc[:,'text']
    y_train=df.loc[:,'target']
    X_test=df_Xtest.loc[:,'text']
    y_test=df_ytest.loc[:,'target']

    X_train=preprocess(X_train)
    X_test=preprocess(X_test)
    X_train, X_test=vectorize(X_train, X_test, min_tdf)
    return X_train, y_train, X_test, y_test
