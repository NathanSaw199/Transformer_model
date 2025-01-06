import pandas as pd
import re
import string,time
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import emoji
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import spacy
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv(r"C:\Users\Saw\Desktop\GEN_AI\IMDB Dataset.csv")

# print(df.shape)
# print(df.head())
df = df.head(100)
# print(df.shape)
# print(df['review'][3])
# df = df['review'] [4]
# print(df)

#STEP 1 >>>>> make lower case
df['review'] = df['review'].str.lower()
# print(df['review'][3])

#STEP 2 >>>>> remove HTML tags
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)

# text = "<html><body><p> Movie 1</p><p> Actor - Aamir Khan</p><p> Click here to <a href='http://google.com'>download</a></p></body></html>"

# print(remove_html_tags(text))

df['review'] = df['review'].apply(remove_html_tags)
# print(df['review'][10])

#STEP 3 >> remove URL

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

# text1 = 'Check out my youtube https://www.youtube.com/dswithbappy dswithbappy'
# text2 = 'Check out my linkedin https://www.linkedin.com/in/boktiarahmed73/'
# text3 = 'Google search here www.google.com'
# text4 = 'For data click https://www.kaggle.com/'

# print(remove_url(text1))


#STEP 4 >>>>> punctuation handling 

exclude = string.punctuation
def remove_punc(text):
    for char in exclude:
        text = text.replace(char,'')
    return text
# text = 'string. With. Punctuation?'
# start = time.time()
# print(remove_punc(text))
# time1 = time.time() - start
# print(time1*50000)

def remove_punc1(text):
    return text.translate(str.maketrans('','',exclude))

# start = time.time()
# remove_punc1(text)
# time2 = time.time() - start
# print(time2*50000)


# print(remove_punc1(df['review'][5]))

chat_words = {
    'AFAIK':'As Far As I Know',
    'AFK':'Away From Keyboard',
    'ASAP':'As Soon As Possible',
    "FYI": "For Your Information",
    "ASAP": "As Soon As Possible",
    "BRB": "Be Right Back",
    "BTW": "By The Way",
    "OMG": "Oh My God",
    "IMO": "In My Opinion",
    "LOL": "Laugh Out Loud",
    "TTYL": "Talk To You Later",
    "GTG": "Got To Go",
    "TTYT": "Talk To You Tomorrow",
    "IDK": "I Don't Know",
    "TMI": "Too Much Information",
    "IMHO": "In My Humble Opinion",
    "ICYMI": "In Case You Missed It",
    "AFAIK": "As Far As I Know",
    "BTW": "By The Way",
    "FAQ": "Frequently Asked Questions",
    "TGIF": "Thank God It's Friday",
    "FYA": "For Your Action",
    "ICYMI": "In Case You Missed It",
}

def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

# print(chat_conversion("DO this work FYI"))

#STEP 5 >>>>> incorrect text handling 
incorrect_text = 'ceertain conditionas duriing seveal ggenerations aree moodified in the saame maner.'
textBlb = TextBlob(incorrect_text)
print(textBlb.correct().string)

#STEP 6 >>>>> STOP WORDS
# print(stopwords.words('english'))

# print(len(stopwords.words('english')))

def remove_stopwards(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append(' ')
        else:
            new_text.append(word)
    
    x = new_text[:]
    new_text.clear()
    return " ".join(x)



# print(remove_stopwards('probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it\'s not preachy or boring. it just never gets old, despite my having seen it some 15 or more times'))
# print(df["review"].apply(remove_stopwards))


#STEP 7 >>>>> remove emoji 
# search emoji unicode

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# print(remove_emoji("Loved the movie. It was 😘😘"))
# print(emoji.demojize('Python is 🔥'))

#STEP 8 >>>>> tokenization 
 
# word tokenization
sent1 = 'I am going to delhi'
# print(sent1.split())

# sentence tokenization
sent2 = 'I am going to delhi. I will stay there for 3 days. Let\'s hope the trip to be great'
# print(sent2.split('.'))

# Problems with split function
sent3 = 'I am going to delhi!'
# print(sent3.split())
sent4 = 'Where do think I should go? I have 3 day holiday'
# print(sent4.split('.'))

# STEP 9 >>>>> Regular Expression
sent3 = 'I am going to delhi!'
tokens = re.findall("[\w']+", sent3)
# print(tokens)


text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry?
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
when an unknown printer took a galley of type and scrambled it to make a type specimen book."""
sentences = re.compile('[.!?] ').split(text)
# print(sentences)

sent1 = 'I am going to visit delhi!'
# print(word_tokenize(sent1))

text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry?
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
when an unknown printer took a galley of type and scrambled it to make a type specimen book."""

# print(sent_tokenize(text))

sent5 = 'I have a Ph.D in A.I'
sent6 = "We're here to help! mail us at nks@gmail.com"
sent7 = 'A 5km ride cost $10.50'

print(word_tokenize(sent5))
print(word_tokenize(sent6))


# SPACY GOOD 
nlp = spacy.load('en_core_web_sm')
doc1 = nlp(sent5)
doc2 = nlp(sent6)
doc3 = nlp(sent7)
doc4 = nlp(sent1)
# print(doc4)

# for token in doc4:
#     print(token)

#STEMMER

ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

sample = "walk walks walking walked"
# print(stem_words(sample))

text = 'probably my alltime favorite movie a story of selflessness sacrifice and dedication to a noble cause but its not preachy or boring it just never gets old despite my having seen it some 15 or more times in the last 25 years paul lukas performance brings tears to my eyes and bette davis in one of her very few truly sympathetic roles is a delight the kids are as grandma says more like dressedup midgets than children but that only makes them more fun to watch and the mothers slow awakening to whats happening in the world and under her own roof is believable and startling if i had a dozen thumbs theyd all be up for this movie'
# print(text)

# print(stem_words(text))

#LEMMATIZATION
#: Stemming & lamatization are same to retrieve root words but lamatization is worked good. Lamatization is slow & stemming is fast
wordnet_lemmatizer = WordNetLemmatizer()
sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

print(sentence_words)

print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos='v')))