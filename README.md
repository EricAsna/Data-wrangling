# Twitter Archive Data Wrangling
WeRateDogs is a Twitter account that rates people’s dogs with a humorous comment about the dog. In this project, the wrangling efforts are conducted on the tweet archive of WeRateDogs account using Python and its libraries.
## Data Gathering
Data has been gathered from the following sources:

1.	The WeRateDogs twitter archive which is given to be downloaded manually and uploaded to the jupyter notebook.
2.	The tweet image predictions that contains the breed of dogs identiﬁed from dog images bu running every image through a neural network. This data was downloaded programmati-cally using the requests library.
3.	The Twitter API for each tweet’s JSON data using Python’s Tweepy library and store each tweet’s entire set of JSON data in a ﬁle. The outputs of the twitter API are in json and these outputs were stored in a text ﬁle called tweet_json.txt in separate lines. Deleted tweets were also stored in a dictionary. After storing the data, the text ﬁle was opened and read to extract tweet_id, retweet_counts and favorite_counts from each json content.
## Data Source
All data are available in Source folder.
