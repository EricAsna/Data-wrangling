#!/usr/bin/env python
# coding: utf-8

# # 1. Gathering Data

# In[48]:


import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
#matplotlib inline

df = pd.read_csv('twitter-archive-enhanced.csv')


# In[49]:


# Downloading image prediction file programmatically
r = requests.get('https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv')
with open('image_predictions' + '.tsv', 'wb') as f:
    f.write(r.content)


# In[50]:


df_breed= pd.read_csv('image_predictions.tsv', sep = '\t')


# In[4]:


#Twitter API data

import tweepy
import json

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

id_of_tweet = df.tweet_id

errors = {}
with open('tweet_json.txt', 'w') as outfile:
    for i in id_of_tweet:
        try:
            ranking = np.where(id_of_tweet == i)[0][0]
            print(ranking)
            tweet = api.get_status(i, tweet_mode = 'extended')
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except Exception as e:
            errors[str(ranking)] = i

           


# In[51]:


# Reading the stored json file & creating a pandas Dataframe

import json
with open('tweet_json.txt' , 'r') as json_file:  
     json_data = [json.loads(line) for line in json_file]
        
df_list = []
for i in range(len(json_data)):
    df_list.append({'tweet_id': json_data[i]['id'],
                     'retweet_count': json_data[i]['retweet_count'],
                     'favorite_count': json_data[i]['favorite_count']})

df_tweet = pd.DataFrame(df_list, columns = ['tweet_id', 'retweet_count', 'favorite_count'])


# In[ ]:


# Deleted tweets
deleted_tweets = errors


# # 2. Assessing Data

# In[52]:


df.head(10)


# In[53]:


df.info()


# In[54]:


# Incorrect identified dog names 
df[[x.islower() for x in df.name]].name.value_counts()


# In[55]:


df.rating_denominator.value_counts()


# In[56]:


df_breed.head(10)


# In[57]:


df_breed.query('p1_dog == False and p2_dog == False and p3_dog == False')


# In[58]:


df_breed.info()


# In[59]:


df_tweet.head(10)


# In[60]:


df_tweet.info()


# In[61]:


df_tweet.retweet_count.min(), df_tweet.favorite_count.min()


# ## 2.1. Quality 
# 
# - df: dog names column has missing values (None instead of NaN) and incorrect names ('a', 'the', 'very', etc.).
# - df: "in-reply_to_status_id" and "in_reply_to_user_id" and "source" columns are unnecessary for our analysis.
# - df: "retweeted_status_id", "retweeted_status_user_id" and "retweeted_status_timestamp" columns are related to retweets.
# - df: Missing values in dog stages columns are filled with None instead of NaN.
# - df: Dog stages have object datatype.
# - df: Timestamp column has object datatype.
# - df: The text column contains the url. Expanded_url also has that information.
# - df_breed: Breed name of dogs start with capital and small letters.
# - df_breed: There are rows in p1-dog, p2-dog, p3-dog columns with False values.
# - df, df_breed and df_tweet are of different length. Image predictions are available until 1 August 2017.
# - expanded_urls includes JPEG images url as well.
# 
# 
# 
# ## 2.2. Tidiness
# 
# - df: Dog stages are in 4 different columns.
# - df_breed: It has 9 columns for dog's breed, confidence and dog prediction.
# - Master table 
# 

# # 3. Cleaning Data

# In[62]:


# Take copy of each dataframe before starting the cleaning process
df_name_clean = df.copy()
df_breed_clean = df_breed.copy()
df_tweet_clean = df_tweet.copy()


# ## Define
# Replace None values and incorrect names in dog names with NaN in df Dataframe.  

# ## Code

# In[63]:


df_name_clean['name'] = df_name_clean.name.replace('None', np.nan)
incorr_names = df_name_clean[[x.islower() for x in df.name]].name
df_name_clean['name'] = df_name_clean.name.replace(incorr_names, np.nan)


# ## Test

# In[64]:


df_name_clean[[x.islower() for x in df.name]].name.value_counts()


# ## Define
# Drop unneccesary columns from df_name_clean

# ## Code

# In[65]:


df_name_clean.drop(['in_reply_to_status_id', 'in_reply_to_user_id', 'source'], axis = 1, inplace = True)


# ## Test

# In[66]:


df_name_clean


# ## Define
# Removing rows relating to retweets from df_name_clean and then dropping the retweeted columns.

# ## Code

# In[67]:


retweet_indices = df_name_clean[df_name_clean.retweeted_status_id.notnull()].index.values
df_name_clean.drop(retweet_indices, axis = 0, inplace = True)
df_name_clean.drop(['retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], axis = 1, inplace = True)


# ## Test

# In[68]:


df_name_clean


# ## Define
# Replace None values in dog stages with NaN.

# ## Code

# In[69]:


df_name_clean[['doggo', 'floofer', 'pupper', 'puppo']] = df_name_clean[['doggo', 'floofer', 'pupper', 'puppo']].replace('None', np.nan)


# ## Test

# In[70]:


df_name_clean


# ## Define
# Creating one column representing the dog stage instead of 4.

# ## Code

# In[71]:


doggo_indices = df_name_clean[df_name_clean['doggo'].notnull() == True].index.values
floofer_indices = df_name_clean[df_name_clean['floofer'].notnull() == True].index.values
pupper_indices = df_name_clean[df_name_clean['pupper'].notnull() == True].index.values
puppo_indices = df_name_clean[df_name_clean['puppo'].notnull() == True].index.values


# In[72]:


array = np.repeat(np.nan, len(df_name_clean))
df_name_clean['dog_stage'] = array
df_name_clean['dog_stage'].loc[doggo_indices] = 'doggo'
df_name_clean['dog_stage'].loc[floofer_indices] = 'floofer'
df_name_clean['dog_stage'].loc[pupper_indices]= 'pupper'
df_name_clean['dog_stage'].loc[puppo_indices]= 'puppo'


# Dropping the extra columns related to dog stage

df_name_clean.drop(['doggo', 'floofer', 'pupper', 'puppo'], axis = 1, inplace = True)
df_name_clean = df_name_clean.reset_index(drop = True)


# ## Test

# In[73]:


df_name_clean.head(50)


# ## Define
# Change the datatype of dog_stage and timestamp columns to 'category' and 'datetime' respectively.

# ## Code

# In[74]:


df_name_clean.dog_stage.astype('category')
df_name_clean.timestamp = pd.to_datetime(df_name_clean.timestamp)


# ## Test

# In[75]:


df_name_clean.info()


# ## Define
# Remove the url from the text column since it is already provided in the expanded_url column.

# ## Code

# In[76]:


df_name_clean['text'] = df_name_clean.text.apply(lambda x: x.split('https')[0])


# ## Test

# In[77]:


df_name_clean.text[0]


# ## Define
# Identifying the breed of dog and create a single column for that and placing NaN for False values for all the three predictions (and removing the extra columns).

# ## Code

# In[78]:


def dog_breed(dog):
    if dog['p1_dog'] == True:
        dog_breed = dog['p1']
        p_conf = dog['p1_conf']
    elif dog['p2_dog'] == True:
        dog_breed = dog['p2']
        p_conf = dog['p2_conf']
    elif dog['p3_dog'] == True:
        dog_breed = dog['p3']
        p_conf = dog['p3_conf']
    else:
        dog_breed = np.nan
        p_conf = np.nan
    return pd.Series([dog_breed, p_conf])
        


# In[79]:


df_breed_clean[['dog_breed', 'dog_breed_conf']] = df_breed_clean.apply(dog_breed, axis = 1)
df_breed_clean.drop(['p1', 'p2', 'p3', 'p1_conf', 'p2_conf', 'p3_conf', 'p1_dog', 'p2_dog', 'p3_dog'], axis = 1, inplace = True)


# ## Test

# In[80]:


df_breed_clean.head(10)


# ## Define
# Uppercase the first letter of the dog_breed names.

# ## Code

# In[81]:


df_breed_clean['dog_breed'] = df_breed_clean.dog_breed.str.capitalize()


# ## Test

# In[82]:


df_breed_clean.head()


# ## Define
# Create a master tables containing all the values (since all tables are related to one another).

# ## Code

# In[83]:


twitter_archive_master = pd.merge(df_name_clean, df_breed_clean, on = ['tweet_id'], how = 'inner')
twitter_archive_master = pd.merge(twitter_archive_master, df_tweet, on = ['tweet_id'], how = 'inner')


# ## Test

# In[84]:


twitter_archive_master.head()


# ## Define
# Drop unnecessary columns from the master table.

# In[85]:


twitter_archive_master.drop(['jpg_url', 'img_num'], axis = 1, inplace = True)


# In[86]:


twitter_archive_master


# ## Define
# Change the order of columns in the master table. Also change column "name" to "dog_name".

# ## Code

# In[87]:


col = twitter_archive_master.columns.tolist()
col


# In[88]:


col = col[0:4] + col[6:10] + col[4:6] + col[10:12]
col


# In[89]:


twitter_archive_master = twitter_archive_master[col]
twitter_archive_master = twitter_archive_master.rename(columns = {'name' : 'dog_name'})


# ## Test

# In[90]:


twitter_archive_master


# # 4. Data visualization

# In[91]:


# Showing the relationship between the number of favorites and retweets
plt.scatter(twitter_archive_master['retweet_count'], twitter_archive_master['favorite_count']);
plt.xlabel('Retweet_Counts');
plt.ylabel('Favorite_Counts');
plt.title('WeRateDogs - Retweets vs. Favorites');


# In[94]:


pd.concat([
    pd.concat([twitter_archive_master, df_name_clean, df_breed_clean, df_tweet_clean], axis=1)]).to_csv('twitter_archive_master.csv', index = False)

