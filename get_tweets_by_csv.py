import tweepy
from tweepy import OAuthHandler
import os
import pandas as pd
import tqdm
class Spdder_Twitter(object):
    def __init__(self):

        consumer_key = "RafkuPTfkFEn7DjPExBmP7jn5"
        consumer_secret = "ubYdNV1vqi8e1uEEnlbunP8AcwqkxtEQtIZZN3tbrOCcWwO1V5"
        access_token = "1086665342628388864-YV6M2oKaQfWxPy8P02BkmHaYLGFd80"
        access_token_secret = "CD1269b8yaykUI5dKz8KDLeYZWeFeScNxvUZIMsreGAXm"

        #auth = OAuthHandler(consumer_key, consumer_secret)
        #auth.set_access_token(access_token, access_token_secret)


        #self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    def get_tweets_by_csv(self):
        output_path='datasets/public/w+ws/'
        csv_file=pd.read_csv(output_path+'labeled_data.csv')

        tweets_ids = list(csv_file['id'].values)
        for tweet_id in tweets_ids:
            if os.path.exists(output_path+'txts/'+str(tweet_id)+'.txt'):
                tweets_ids.remove(tweet_id)
                print('remove ',tweet_id)
        tweet_num=len(tweets_ids)
        tweet_index=0
        while tweet_index< tweet_num:
            print('index: %d/%d' % (tweet_index, tweet_num))

            tweet_batch=tweets_ids[tweet_index:tweet_index+99]
            tweet_index+=len(tweet_batch)
            #public_tweets = self.api.statuses_lookup(tweet_batch)
            public_tweets=tweet_batch
            for tag,context in zip(tweet_batch,public_tweets):
                print(output_path+'txts/'+str(tag)+'.txt',context)
                '''with open(output_path+'txts/'+str(tag)+'.txt','w') as f:
                    f.write(str(public_tweets))'''





#twitter_contain=Spdder_Twitter()
#twitter_contain.get_tweets_by_csv()
output_path='datasets/public/w+ws/'
csv_file=pd.read_csv(output_path+'labeled_data.csv')
print(csv_file['id'])
for i in range(len(csv_file['id'])-1,-1,-1):
    print(i)
    data=csv_file['id'][i]
    if not os.path.exists(output_path+'txts/'+str(data)+'.txt'):
        print('remove ',data)
        csv_file.drop(i,inplace=True)
tweets=[]
for i in csv_file['id']:
    with open(output_path+'txts/'+str(i)+'.txt','r') as f:
        tweets.append(f.read().strip())
csv_file.insert(0,'tweet',tweets)
csv_file.to_csv(output_path+'new_label_data.csv')
print(csv_file)
