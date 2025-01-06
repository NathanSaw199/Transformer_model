
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import gensim
from gensim.utils import simple_preprocess


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 255)
from nltk import sent_tokenize

df = pd.read_csv(r"C:\Users\Saw\Desktop\GEN_AI\IMDB Dataset.csv")
# print(df.head())
# print(df.shape)
df = df.iloc[:10000]
# print(df.shape)
# print(df['review'][0])
# print(df['sentiment'].value_counts())

#missing value
# print(df.isnull().sum())

#duplicate value check
# print(df.duplicated().sum())
# df.drop_duplicates(inplace=True)
# print(df.duplicated().sum())

#Remove tags - HTML
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text
df['review'] = df['review'].apply(remove_tags)

# print(df.head())
#df['review'][0]

#make it lower 
df['review'] = df['review'].apply(lambda x:x.lower())
# print(df['review'][0])
nltk.download('stopwords')
sw_list = stopwords.words('english')

df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
# print(df['review'][0])

X = df.iloc[:,0:1]
y = df['sentiment']
# print(X.head())
# print(y.head())
# encode target class to 0 and 1
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
# print(X_train.shape)
# print(X_test.shape)

#applying bags of words
cv = CountVectorizer()

X_train_bow = cv.fit_transform(X_train['review']).toarray() # train data for fit transform
X_test_bow = cv.transform(X_test['review']).toarray() # test data for transform
# print(X_train_bow) 
gnb = GaussianNB()
# gnb.fit(X_train_bow,y_train)

# y_pred = gnb.predict(X_test_bow)

# print(accuracy_score(y_test,y_pred))

# print(confusion_matrix(y_test,y_pred))

rf = RandomForestClassifier()
# rf.fit(X_train_bow,y_train)
# y_pred = rf.predict(X_test_bow)

# print(accuracy_score(y_test,y_pred))

cv = CountVectorizer(max_features=3000) # pick only 3000 unique words out of the corpus most frequent words

X_train_bow_cv = cv.fit_transform(X_train['review']).toarray()
X_test_bow_cv = cv.transform(X_test['review']).toarray()

rf = RandomForestClassifier()
# rf.fit(X_train_bow_cv,y_train)
# y_pred = rf.predict(X_test_bow_cv)
# print(accuracy_score(y_test,y_pred))

#Ngrams 
cv = CountVectorizer(ngram_range=(2,2),max_features=5000)
X_train_bow_cv = cv.fit_transform(X_train['review']).toarray()
X_test_bow_cv = cv.transform(X_test['review']).toarray()
rf = RandomForestClassifier()
# rf.fit(X_train_bow_cv,y_train)
# y_pred = rf.predict(X_test_bow_cv)
# print(accuracy_score(y_test,y_pred))


#Tfid
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train['review']).toarray()
X_test_tfidf = tfidf.transform(X_test['review'])
rf = RandomForestClassifier()

# rf.fit(X_train_tfidf,y_train)
# y_pred = rf.predict(X_test_tfidf)
# print(accuracy_score(y_test,y_pred))

print(df.head())
#word2vec 
# Tokenize reviews
