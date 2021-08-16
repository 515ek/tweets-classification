from flask import Flask, app
from flask import render_template, request
from flask import redirect
from flask.helpers import url_for
import tweepy_utils, ml_utils

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')

@app.route('/')
def get_index_page():
    twitter_trends = {}
    # print("Starting twitter api")
    global twitter_api
    twitter_api = tweepy_utils.TwitterClient()
    # print(twitter_api)
    twitter_trends['Global'] = twitter_api.get_trends(place=1)
    twitter_trends['India'] = twitter_api.get_trends(place=23424848)

    return render_template('index.html', twitter_trends = twitter_trends)

@app.route('/search', methods=["GET", "POST"])
def get_search_page():
    
    if request.method == "POST":
        req = request.form
        try:
            skey = req.get('search_key')
            if skey:
                print(skey)
                skey = skey + ',' + skey
        except KeyError:
            pass
        return redirect(url_for('classify_tweets', skey = skey))
    return render_template('index.html')

@app.route('/result/<string:skey>')
def classify_tweets(skey):
    # print(skey)
    skey,sname = skey.split(',')
    result = {}
    # twitter_api = tweepy_utils.TwitterClient()
    tweets = twitter_api.get_tweets(skey, r_type='mixed')
    # for tweet in tweets:
    #     print(tweet, '\n')
    classification_report = ml_utils.classify(tweets)
    ml_utils.plot_wordcloud(tweets)
    # result['tweets'] = tweets
    result['sname'] = sname
    result['ntweets'] = len(tweets)
    result['classification_report'] = classification_report

    return render_template('result.html', result = result)

@app.route('/modeleval/')
def eval_model():
    result = {}
    result = ml_utils.evaluate_model()
    return render_template('model_eval.html', result = result)

@app.route('/about/')
def about():
    result = {}
    try:
        with open('static/about.txt', 'r') as fp:
            result['about'] = fp.read()
    except Exception as e:
        print("Error: \n", str(e))
    return render_template('about.html', result = result)

if __name__ == '__main__':
    app.run( debug= True)