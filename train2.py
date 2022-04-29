from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from keras.models import Sequential
from nltk.tokenize import RegexpTokenizer
from pandas.core.reshape.reshape import get_dummies
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re
import nltk
import string
import warnings
import pandas as pd
plt.style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings("ignore")

df = pd.read_excel('data/train/Tweets_train2.xlsx')
df = df[["Sentiment", "Text"]]

# Changing the value of the column "Sentiment" to 0 if the value is "Negatif".
df["Sentiment"][df["Sentiment"]== "Negatif"] = 0
df["Sentiment"][df["Sentiment"]== "Pozitif"] = 1

# Filtering the dataframe by the value of the column "Sentiment" and it is getting the rows that have the value of 1.
df_positive = df[df["Sentiment"] == 1]
df_negative = df[df["Sentiment"] == 0]  

# Getting the first 2000 rows of the dataframe.
df_negative = df_negative.iloc[:int(2000)]
df_positive = df_positive.iloc[:int(2000)]

# Concatenating the dataframes.
df = pd.concat([df_positive, df_negative])
df["Text"] = df["Text"].str.lower() 

# Cleaning and removing stopwords from list
stop_words = list(stopwords.words('turkish'))
", ".join(stopwords.words('turkish'))
STOPWORDS = set(stopwords.words('turkish'))
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df['Text'] = df["Text"].apply(lambda text: cleaning_stopwords(text))

# Cleaning and removing punctuations
turkish_punctuations = string.punctuation
punctuations_list = turkish_punctuations
def cleaning_punctutations(text):
    translator = str.maketrans("", "", punctuations_list)
    return text.translate(translator)
df["Text"] = df["Text"].apply(lambda x: cleaning_punctutations(x))

# Cleaning and removing repeating characters
def cleaning_repeating_char(text):
    return re.sub(r"(.)\1+", r"\1", text)
df["Text"] = df["Text"].apply(lambda x: cleaning_repeating_char(x))

# Cleaning and removing numeric numbers
def cleaning_numbers(data):
    return re.sub("[0-9]", "", str(data))
df["Text"] = df["Text"].apply(lambda x: cleaning_numbers(x))

# Getting Tokenization of tweet text
tokenizer = RegexpTokenizer(r"\w+")
df["Text"] = df["Text"].apply(tokenizer.tokenize)

# Applying stemming
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
df["Text"] = df["Text"].apply(lambda x: stemming_on_text(x))

# Applying lemmatizer
lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
df["Text"] = df["Text"].apply(lambda x: lemmatizer_on_text(x))

# Getting the 3000 most frequent words in the text.
X_train_verb, X_test_verb, y_train_verb, y_test_verb = train_test_split(df.Text, df.Sentiment, test_size=0.2, random_state=0)
tokenizer = Tokenizer(num_words=3000, split=",")
tokenizer.fit_on_texts(df.Text.values)
sequences = tokenizer.texts_to_sequences(df.Text.values)
X = pad_sequences(sequences, padding="post")

model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(Dropout(0.8))
model.add(LSTM(196))
model.add(Dense(2, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

y = get_dummies(df.Sentiment).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train, epochs=8, batch_size=16, verbose=2)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
#[print(X_test_verb.values[i], y_pred[i], y_test[i]) for i in range(len(y_test))]
#accr1 = model.evaluate(X_train,y_train) #we are starting to test the model here
accr1 = accuracy_score(y_test, y_pred)
print(accr1)

def predict(tweet):
    tokenizer = Tokenizer(num_words=300, split=",")
    tokenizer.fit_on_texts(tweet)
    sequences = tokenizer.texts_to_sequences(tweet)
    Q = pad_sequences(sequences, padding="post", maxlen=X.shape[1])
    predictions = model.predict(Q)
    predictions = (predictions > 0.5)
    if predictions[0][0] == True:
        return "Negative"
    else:
        return "Pozitive"
