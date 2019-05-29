#  -*- coding: utf-8 -*-
# 利用tweepy API爬取
import tweepy
import time
import json
from tweepy import OAuthHandler
import re
import os
import logging

logging.basicConfig()

'''dict = {}
L = []
with open('label1.txt', 'r') as f:
    lines = f.read().splitlines()
    for i in lines:
        # print(i)
        line = re.split(":", i)
        L.append(line[1])
        dict[line[1]] = line[0]'''
# print("列表L为：",L)
# print("字典dict为：",dict)


consumer_key = "RafkuPTfkFEn7DjPExBmP7jn5"
consumer_secret = "ubYdNV1vqi8e1uEEnlbunP8AcwqkxtEQtIZZN3tbrOCcWwO1V5"
access_token = "1086665342628388864-YV6M2oKaQfWxPy8P02BkmHaYLGFd80"
access_token_secret = "CD1269b8yaykUI5dKz8KDLeYZWeFeScNxvUZIMsreGAXm"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

origin_result = []

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#api = tweepy.API(auth,proxy="http://mg.520ssr.ga:1080")
tweet1=api.retweets("498430783699554305",200)
print(tweet1)
# re_tweet_ = api.retweets(id)
# print(tweet1[0])
# 获取某人的微博
# api.get_user('用户名').timeline()
'''for i in range(40):
    origin_tweet = api.statuses_lookup(L[100:150])
    # print(len(origin_tweet))
    for t in origin_tweet:
        count = 0
        # api.statuses_lookup([t.id_str])
        origin_result.append({
            'label': dict[t.id_str],
            'is_quote_status': t.is_quote_status,
            'user_geo_enabled': t.user.geo_enabled,
            'user_created_at': str(t.user.created_at),
            'verified': t.user.verified,
            'statuses_count': t.user.statuses_count,
            'location': t.user.location,
            'friends_count': t.user.friends_count,
            'followers_count': t.user.followers_count,
            'favorite_count': t.favorite_count,
            'retweet_count': t.retweet_count,
            'text': t.text,
            'user_name': t.user.screen_name,
            'tweet_created_at': str(t.created_at),
            'tweet_id': t.id_str,
            'user_id': t.user.id,
            'user_description': t.user.description
        })

        re_tweets = api.retweets(t.id_str, 200)
        for tweet in re_tweets:
            # time.sleep(14)
            # print("休眠中")
            origin_result.append({
                'is_quote_status': tweet.is_quote_status,
                'user_geo_enabled': tweet.user.geo_enabled,
                'user_created_at': str(tweet.user.created_at),
                'verified': tweet.user.verified,
                'statuses_count': tweet.user.statuses_count,
                'location': tweet.user.location,
                'friends_count': tweet.user.friends_count,
                'followers_count': tweet.user.followers_count,
                'favorite_count': tweet.favorite_count,
                'retweet_count': tweet.retweet_count,
                'text': tweet.text,
                'user_name': tweet.user.screen_name,
                'tweet_created_at': str(tweet.created_at),
                'tweet_id': tweet.id_str,
                'user_id': tweet.user.id,
                'user_description': tweet.user.description
            })
            count = count + 1
        with open(os.path.join("tweet15", t.id_str + ".json"), 'w+') as f:
            json.dump(origin_result, f, indent=4)
        print("Event :", len(origin_result))
        origin_result[:] = []
    # print("速率限制，休眠中")
    # time.sleep(15*60)
    break
print("\n")
print("Total: ", len(os.listdir("tweet15")))'''

# for tweet in tweet1:
#     # quote_tweet=api.statuses_lookup([tweet.id_str])
#     result.append({
#         'is_quote_status':tweet.is_quote_status,
#         'user_geo_enabled': tweet.user.geo_enabled,
#         'user_created_at': str(tweet.user.created_at),
#         'verified': tweet.user.verified,
#         'statuses_count': tweet.user.statuses_count,
#         'location': tweet.user.location,
#         'friends_count': tweet.user.friends_count,
#         'followers_count': tweet.user.followers_count,
#         'favorite_count': tweet.favorite_count,
#         'retweet_count': tweet.retweet_count,
#         'text': tweet.text,
#         'user_name': tweet.user.screen_name,
#         'tweet_created_at':str(tweet.created_at),
#         'tweet_id':tweet.id_str,
#         'user_id':tweet.user.id,
#         'user_description':tweet.user.description
#     })
# print(t.coordinates)
# print(tweet)

# print(len(result))
# public_tweets = api.user_timeline(691809004356501505)
# public_tweets = api.statuses_lookup([691809004356501505])
