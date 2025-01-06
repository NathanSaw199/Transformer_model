import numpy as np
import pandas as pd
import gensim
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from sklearn.decomposition import PCA
import plotly.express as px

#BAG OF WORDS

df = pd.DataFrame({"texts":["people watch dswithbappy",
                         "dswithbappy watch dswithbappy",
                         "people write comment",
                          "dswithbappy write comment"],
                          "output":[1,1,0,0]})

# print(df)

cv = CountVectorizer()
bow = cv.fit_transform(df['texts'])
# print(cv.vocabulary_)
# print(bow.toarray())
# print(cv.transform(['bappy watch dswithbappy']).toarray())

X = bow.toarray()
y= df['output']

#Ngrams

df = pd.DataFrame({"text":["people watch dswithbappy",
                         "dswithbappy watch dswithbappy",
                         "people write comment",
                          "dswithbappy write comment"],
                          "output":[1,1,0,0]})

cv = CountVectorizer(ngram_range=(2,2))

bow = cv.fit_transform(df['text'])

# print(cv.vocabulary_)

print(bow[0].toarray())
print(bow[1].toarray())
print(bow[2].toarray())

#Ti gram
# BI grams
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(3,3))
bow = cv.fit_transform(df['text'])
# print(cv.vocabulary_)
# print(bow[0].toarray())
# print(bow[1].toarray())
# print(bow[2].toarray())


#TF-IDF (Term frequency- Inverse document frequency)
df = pd.DataFrame({"text":["people watch dswithbappy",
                         "dswithbappy watch dswithbappy",
                         "people write comment",
                          "dswithbappy write comment"],"output":[1,1,0,0]})

tfid = TfidfVectorizer()
arr = tfid.fit_transform(df['text']).toarray()
# print(arr)


#word2vec
# corpus =  """A Game Of Thrones 
#     Book One of A Song of Ice and Fire 
#     By George R. R. Martin 
#     PROLOGUE 
#     "We should start back," Gared urged as the woods began to grow dark around them. "The wildlings are 
#     dead." 
#     "Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile. 
#     Gared did not rise to the bait. He was an old man, past fifty, and he had seen the lordlings come and go. 
#     "Dead is dead," he said. "We have no business with the dead." 
#     "Are they dead?" Royce asked softly. "What proof have we?" 
#     "Will saw them," Gared said. "If he says they are dead, that's proof enough for me." 
#     Will had known they would drag him into the quarrel sooner or later. He wished it had been later rather 
#     than sooner. "My mother told me that dead men sing no songs," he put in. 
#     "My wet nurse said the same thing, Will," Royce replied. "Never believe anything you hear at a woman's 
#     tit. There are things to be learned even from the dead." His voice echoed, too loud in the twilit forest."""


story = []
file_path = r'C:\Users\Saw\Desktop\GEN_AI\001ssb.txt'

with open(file_path,'r',encoding='utf-8') as f:
    corpus = f.read()
    raw_sent = sent_tokenize(corpus) #tokenize into sentences
    for sent in raw_sent:
        story.append(simple_preprocess(sent)) # preprocess each sentences

# print(story[0])

# print(len(story[0]))

model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)

model.build_vocab(story)
# print(model.train(story,total_examples=model.corpus_count,epochs=model.epochs))
# print(model.wv.most_similar('daenerys'))
# print(model.wv.similarity('arya','sansa'))
# pass this to model (important)
vec = model.wv.get_normed_vectors()
# print(vec)

pca = PCA(n_components=3)
X = pca.fit_transform(model.wv.get_normed_vectors())

print(X)

fig = px.scatter_3d(X[200:300],x=0,y=1,z=2, color=y[200:300])
fig.show()