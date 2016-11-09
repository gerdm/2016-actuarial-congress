import twitter
import numpy as np
import pandas as pd
import pickle


# Accessing the Twitter API
api = twitter.Api(consumer_key, consumer_secret, access_token, access_token_secret, sleep_on_rate_limit=True)

def search_debate(max_id):
    return api.GetSearch("Donald OR Trump Or Hillary OR Clinton OR #debate", result_type="recent", since="2016-10-09", until="2016-10-10", max_id=max_id, count=100)

total_tweets = []

# Parse the target dates starting from end all until the iterator is stopped
last_id = None
for step in range(100):
    if last_id is not None:
        results = search_debate(last_id)
        last_id = results[-1].id
    else:
        results = api.GetSearch("Donald OR Trump Or Hillary OR Clinton OR #debate", result_type="recent", since="2016-10-09", until="2016-10-10", count=100)
    
    last_id = results[-1].id
    print(results[-1].created_at, end="\r")

    total_tweets += results
    
print()
print(api.rate_limit.get_limit("GET search/tweets"))
print("Saving to Pickle as 'debate.pickle'")
pickle.dump(total_tweets, open("debate.pickle", "wb"))
