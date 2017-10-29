import tweepy
import csv
import pickle
import numpy


ids=[]
labels=[]



with open("bigdata.csv","r") as file:
    reader=csv.reader(file,delimiter='\t')
    i=0
    for row in reader:
        ids.append(row[0])
        labels.append(row[1])
        if(i==30000):
            break
        i=i+1


l=numpy.unique(labels)
print(l)

dictionary=dict(zip(ids,labels))    #mapping the tweet ids to tweet text.
print(i)


def lookup_tweets(tweet_IDs, api):
    full_tweets = []
    tweet_count = len(tweet_IDs)

    try:
        for i in range(int(tweet_count / 100) + 1):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            full_tweets.extend(
                api.statuses_lookup(id_=tweet_IDs[i * 100:end_loc]))
        return full_tweets
    except tweepy.TweepError:
        print('Something went wrong, quitting...')


api_key = "aYzg08VZzv4RKLCj6f4e698t2"
api_secret = "IzRhirVmKZmEBGSchGQHCQsPCPdA5RYsy1gus1wzYfa2Im4PvK"
access_token_key = "183650922-yz1CcwXGBid6i3ounFzSYWSymkLjOvhBCtrKQjq4"
access_token_secret = "gOJHT57dhn5ue5Dlb6KvdUSKwxp93oJSYjpZK7HOiOIj9"


auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)



results = lookup_tweets(ids, api)
print(len(results))


content=[]
labels=[]
for tweet in results:
    if tweet:
        content.append(tweet.text)
        labels.append(dictionary[str(tweet.id)])
        #print(dictionary[str(tweet.id)])
        

pickle.dump(content,open("content.pkl","wb"))
pickle.dump(labels,open("labels.pkl","wb"))



