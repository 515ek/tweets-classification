import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
import seaborn as sns
import re, string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud, STOPWORDS
import pickle
import time

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords


def plot_wordcloud(tweets):
    try:
        wordcloud = WordCloud(
        background_color= "black",
        width = 1600,
        height= 800,
        random_state= 0,
        collocations= False,
        stopwords= STOPWORDS
        ).generate(' '.join(tweets))

        plt.figure(figsize = (20, 10), facecolor='k')
        wordcloud.to_file('static/ml_files/wordcloud.png')
        return 1
    except Exception as e:
        print("Error: \n", str(e))
        return -1


def preprocess_tweet(tweet):
    tweet = tweet.lower() ## lowercase all the words
    tweet = re.sub(r"@[^\s]+", '', tweet) ## remove usernames
    tweet = re.sub(r'((http://)[^ ]*|(http://)[^ ]*| (www\.)[^ ]*)', '', tweet)
    tweet = re.sub('\n', ' ', tweet)
    for emote in emoticons.keys():
        tweet = re.sub('%s' %re.escape(emote), emoticons[emote], tweet)
    tweet = re.sub('\w*\d\w*', '', tweet)
    tweet = re.sub('<.*?>+', '', tweet)
    tweet = re.sub('[%s]' %re.escape(string.punctuation), '', tweet)
    tweet = re.sub(r'[ ]+', ' ', tweet)  ## convert multiple whitespaces to single
    tweet = re.sub('[^a-z ]*', '', tweet)
    return tweet

def remove_stopwords(text):
    wnet = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    text = ' '.join([wnet.lemmatize(word) for word in text.split(' ') if not word in stop_words])
    return text

def prepare_train_data(train_data):
    start = time.time()
    processed_tweets = train_data['text'].apply(preprocess_tweet)
    processed_tweets = processed_tweets.apply(remove_stopwords)
    end = time.time()
    print("took ", end - start, " seconds to complete preprocessing text")
    le = LabelEncoder()
    Y = le.fit_transform(train_data.target)
    X = np.array(processed_tweets)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.1)
    return (X_train, X_test, y_train, y_test)

def train_model():
    col_names = ['target', 'id', 'date', 'flag', 'user','text']
    train_data = pd.read_csv("data/train_data.csv", header= None, encoding= 'ISO-8859-1', names= col_names)
    # print(len(train_data))
    X_train, X_test, y_train, y_test = prepare_train_data(train_data)
    tfidf_vect = TfidfVectorizer(ngram_range=(1,3))
    X_train = tfidf_vect.fit_transform(X_train)
    # print(X_train.shape,' <<--->>' ,y_train.shape)
    # X_test = tfidf_vect.transform(X_test)
    model = LogisticRegression(
        penalty = 'l2',
        C = 100,
        solver = 'saga',
        random_state = 1,
        n_jobs = 8,
        verbose = 1
    )
    model = model.fit(X_train, y_train)
    # print(model)
    model_variables = {
        'model': model,
        'tfidf_vect': tfidf_vect
    }
    evaluation_variables = {
        'model': model,
        'tfidf_vect': tfidf_vect,
        'X_test': X_test,
        'y_test': y_test
    }
    save_evaluation_variables(evaluation_variables)
    save_classification_model(model_variables)
    return model_variables

def get_classification_model():
    try:
        with open('static/ml_files/model.pkl', 'rb') as fp:
            model_variables = pickle.load(fp)
        return model_variables
    except Exception as e:
        print("Error: \n", str(e))
        return None

def get_evaluation_variables():
    try:
        with open('static/ml_files/eval_variables.pkl', 'rb') as fp:
            evaluation_variables = pickle.load(fp)
        return evaluation_variables
    except Exception as e:
        print("Error: \n", str(e))
        return None

def save_evaluation_variables(evaluation_variables):
    try:
        with open('static/ml_files/eval_variables.pkl', 'wb') as fp:
            pickle.dump(evaluation_variables, fp)
        return 0
    except Exception as e:
        print("Error: \n", str(e))
        return -1

def save_classification_model(model_variables):
    try:
        with open('static/ml_files/model.pkl', 'wb') as fp:
            pickle.dump(model_variables, fp)
        return 0
    except Exception as e:
        print("Error: \n", str(e))
        return -1

def prepare_predict_data(tweets, tfidf_vect):
    try:
        tweets = pd.Series(tweets, name= 'tweets')
        # print(tweets.head())
        tweets = tweets.apply(preprocess_tweet)
        tweets = tweets.apply(remove_stopwords)
        tweets = tfidf_vect.transform(np.array(tweets))
    except Exception as e:
        print("Error: \n", str(e))
    return tweets

def classify(tweets):
    try:
        model_variables = get_classification_model()
        result = []
        if model_variables == None:
            model_variables = train_model()
        model = model_variables.get('model')
        tfidf_vect = model_variables.get('tfidf_vect')
        print(model)
        # print(type(tweets))
        X_pred = prepare_predict_data(tweets, tfidf_vect)
        y_pred = model.predict_proba(X_pred)
        target = y_pred.argmax(axis = 1)
        probs = y_pred.max(axis = 1)
        for i,tweet in enumerate(tweets):
            result.append({'tweet': tweet, 'predict': target[i], 'prob': round(probs[i], 6)})
        return result
    
    except Exception as e:
        print("Error: \n", str(e))
        return None

def evaluate_model():
    result = {}
    evaluation_variables = get_evaluation_variables()
    # print(evaluation_variables)
    try:
        model = evaluation_variables.get('model')
        tfidf_vect = evaluation_variables.get('tfidf_vect')
        X_test = evaluation_variables.get('X_test')
        y_test = evaluation_variables.get('y_test')

        X_test = tfidf_vect.transform(X_test)
        y_pred = model.predict(X_test)

        cls_report = classification_report(y_test, y_pred, output_dict = True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix = conf_matrix / (np.sum(conf_matrix, axis= 1).reshape(2,1))
        plot_confusion_matrix(conf_matrix)
        
        result['cls_report'] = cls_report
        result['conf_matrix'] = conf_matrix

        # print(result)

        return result
    except Exception as e:
        print("Error: \n", str(e))


def plot_confusion_matrix(conf_matrix):
    plt.subplots(figsize = (10,10))
    sns.set(font_scale = 2, style = 'white')
    ax = sns.heatmap(conf_matrix,
                annot= True,
                cmap='RdYlGn',
                linecolor= 'white',
                linewidths= 3
            )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors = 'white', which = 'both')
    plt.xlabel("Predicted Values", fontdict = {'size':16, 'color': 'white' })
    plt.ylabel("Actual Values", fontdict = {'size':16, 'color': 'white' })
    plt.title("Confusion Matrix", fontdict = {'size':24, 'color': 'white' })
    plt.tick_params(colors = 'white', which = 'both' )
    plt.savefig('static/ml_files/eval_confusion_matrix.png', transparent = True)
    # plt.show()
    pass

emoticons = {":-)" : "smile",
             ":)" : "smile",
             ":-(" : "sad smile",
             ":(" : "sad smile",
             ";-)" : "winking smile",
             ";)" : "winking smile",
             ":->" : "grin",
             ":>" : "grin",
             ":-P" : "poke tongue",
             ":P" : "poke tongue",
             ":-/" : "frown",
             ":/" : "frown",
             ":-\\" : "stern face",
             ":\\" : "stern face",
             ":-1" : "smirk",
             ":-D" : "big smile",
             ":D" : "big smile",
             
             "%*}" : "very drunk",
             ":-*" : "kiss",
             "':-)" : "sarcastically raised eyebrow",
             ":-O" : "surprised",
             ":'-(" : "crying",
             ":*(" : "crying",
             
             ":-{)" : "wears a moustache",
             ":-)>" : "wears a beard",
             "%-)" : "wearing spectacles",
             "%)" : "wearing spectacles",
             "&:-\\" : "bad hair day",
             "(:-)" : "bald head", 
             
             "O:-)" : "angel face",
             ">:->" : "devil face",
             "8=X" : "skull and cross bones",
             "(P-|" : "star trek borg",
             ":-[" : "vampire",
             "<:+D" : "clown"
}