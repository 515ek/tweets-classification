from instance.config import API_Key, API_Secret, Access_Token, Access_Token_Secret
from tweepy import OAuthHandler
import tweepy
import json
import pprint
# import instance.config

class TwitterClient(object):
    
    def __init__(self):
        # print(API_Key,API_Secret)
        # print(Access_Token, Access_Token_Secret)
        keys = {
            'api_key': API_Key,
            'api_secret': API_Secret,
            'access_token': Access_Token,
            'access_token_secret': Access_Token_Secret
        }
        # try:
        #     with open("data/twitter-api-keys.dat", 'r') as fp:
        #         data = fp.readlines()
            
        #     data = [line.strip().split('|') for line in data]
        #     keys = {line[0].lower() : line[1] for line in data }
        # except Exception as e:
        #     print("Error: \n ", e)
        
        # print(keys)
        
        try:
            self.auth = OAuthHandler(keys['api_key'], keys['api_secret'])
            
            self.auth.set_access_token(keys['access_token'], keys['access_token_secret'])
            
            self.api = tweepy.API(self.auth)
        except Exception as e:
            print("Error: Authentication Failed:\n", e)
    
    def get_tweets(self, query, count = 20, r_type= 'mixed', tweet_mode = 'extended'):
        
        tweets = []
        
        try:
            fetched_tweets = self.api.search(q = query, count = count, result_type= r_type, tweet_mode = tweet_mode)
            for tweet in fetched_tweets:
                # print(tweet)
                tweets.append(tweet._json['full_text'])
            return tweets
        except tweepy.TweepError as e:
            print("Error : ", +str(e))
            
    def get_trends(self, place = 1):
        try:
            fetch_trends = self.api.trends_place(place)
            fetch_trends = fetch_trends[0]
            trends = fetch_trends.get('trends')
            trends = trends[:5]
            # print(trends)
            # fetch_trends['trends'] = trends
            return trends
        except tweepy.TweepError as e:
            print("Error : ", str(e))