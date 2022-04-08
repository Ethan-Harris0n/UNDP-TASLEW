## TASLEW: Text Analysis and Social Listening for Early Warning

A package built by the UNDP's Crisis Risk and Early Warning team to extract, transform, and analyze text data for Early Warning purposes.


## The following repos provided information and/or source code for many of the functions contained in this package
- https://github.com/MichaelKim0407/tutorial-pip-package - for tutorial on building a python package
- https://github.com/heatherbaier/ - for highlighting said tutorial above
- https://github.com/pewresearch/pewanalytics - several of their co-iccurences / mutual information funcs (modified versions sourced from stack overflow) have been "canabalized" and wrapped under the vectorizer class - chose to do this instead of importing their package due to a number of install issues related to their build 
- https://github.com/dmbeskow/twitter_col - wrapped a number of his excellent functions into a twitter module for ad-hoc data cleaning during data-extraction tasks on databricks

### Summary of methods

An itemized list of methods can be found in the tables below. Many of these functions take a dataframe and column of text as inputs and output a dataframe constituting of the original dataframe's index and outputted column(s) produced by the method that was called.


#### Generic Text DataFrame Methods
.
| Function | Description |
|----------|-------------|
| `prep_text_df`  | Preps a dataframe for analysis by reducing it to only its unique ID and text to be analyzed |
| `get_original_df`  | Merges current text dataframe to original dataframe based on a common ID |

| `find_related_keywords`  | Takes keywords or a list of keywords as input and produces a list of related keywords - useful for unsupervised lexicon building and expansion|
| `make_word_cooccurrence_matrix` | Generates a co-iccurence matrix for a inputted document - useful for unsupervised lexicon building and expansion.


#### General Text Analysis
##### Topic Modeling
Generate topic-analysis of text using a variety of methods (Latent-Derilicht Allocation, Correlation Explanation, Bert, etc.) |
##### Text Classification
This is really just one method but added it as a module to improve ease in application. Zero Shot is an implementation of Hugging Face's Zero Shot Classifcation framework originating from this blog post. 
##### Sentiment Analysis
Calculate Sentiment via a number of frameworks including Facebook's Vader, Flair,etc.  
##### Hate-Speech Classification
Module devoted to Hate-Speech Classification. Currently this only includes Detoxify from Unity-AI's framework.


| func | Description |
|----------|-------------|
| `get_keywords_tfidf`  | Extracts most relevant keywords from datafram using TF-IDF  |
| `get_ngrams`  | Extracts most relevant ngrams from dataframe using TF-IDF  |
| `zero_shot`  | Hugging face pipeline implementation for unsupervised classification of text |
| `get_topics`  | Generate topic-analysis of text using a variety of methods (Latent-Derilicht Allocation, Correlation Explanation, Bert, etc.) |
| `get_sentiment`  | Calculate the sentiment of tweets using Facebook's Vadar or Flair|
| `Identify_HateSpeech`  | Classify probability of text constuting hate-speech using unityAI's detoxify algorithim|


#### Databricks
A few helper functions used for our work on Databricks. These include helpers for interacting with APIs and writing data to Delta Tables.

| Function | Description |
|----------|-------------|
| `get_last_id(table)`  | gets most recent tweet ID from specified delta table. Used automate scheduled API calls for data updation |
| `get_last_timestamp(table)` | Extracts timestamp from most recent facebook post from specified delta table. Used to automate scheduled API calls for data updation  |
| `convert_dates`  | This function converts twitter dates to python date objects  |
| `dedupe_twitter` |  This function dedupes a list of tweets based on tweet ID.  |



#### Twitter Manipulation Methods
| Function | Description |
|----------|-------------|
| `check_tweet`  | Takes user objects and reverses them to create status objects  |
| `combine_dedupe` | Combine and Dedupe list of Twitter JSON Files  |
| `convert_dates`  | This function converts twitter dates to python date objects  |
| `dedupe_twitter` |  This function dedupes a list of tweets based on tweet ID.  |
| `extract_coordinates` | Extracts Geo Coordinates to Pandas Dataframe  |
| `extract_emoji` | Creates  csv containing all emojis in a set of tweets  |
| `extract_gender` | This function will try to guess the gender of the Tweet user based on name. This function uses the gender_guesser package.  |
| `extract_hash_comention` |     Creates hashtag edgelist (either user to hashtag OR comention).  |
| `extract_hashtags` | Creates hashtag edgelist (either user to hashtag OR comention). |
| `extract_media`  |  Creates  csv containing all URLS in a set of tweets |
| `extract_mentions`  |  Creates mention edgelist.  Can return data.frame or write to csv. |
| `extract_reply_network`  |  Creates reply edgelist.  Can return data.frame or write to csv. |
| `extract_retweet_network`  |  Creates retweet edgelist.  Can return data.frame or write to csv. |
| `extract_urls`  |  Creates  csv containing all URLS in a set of tweets |
| `fetch_profiles`  |  A wrapper method around tweepy.API.lookup_users that handles the batch lookup of screen_names.  Returns list of users. |
| `fetch_profiles_file`  |  A wrapper method around tweepy.API.lookup_users that handles the batch lookup of screen_names (saving to file as it goes).  This is better for long lists with memory constraints. |
| `filter_tweets_by_date`  |  Filters tweets in a single file or list of files so that they fall within a given time window.   |
| `get_all_network_files`  |  This is a single command to get the hashtag network, comention network, and retweet network |
| `get_all_tweets`  |  Gets most recent 3240 tweets for a given user. Returns list of tweets. |
| `get_edgelist_file`  |  Builds an agent x agent edgelist of a Tweet json (normal or gzipped) |
| `get_edgelist_from_list`  |  Builds an agent x agent edgelist of a tweet list. |
| `get_emojis`  |  Returns list of emoji's for a tweet.  If no emoji's, returns empty list |
| `get_empty_status()`  |  This function returns an empty or Null status.  This is used to attach to the dictionary of any account that has never tweeted |
| `get_followers_for_id`  |  Gets ALL follower IDS for a given user Adapted from Mike K's code. |
| `get_friend_follower_edgelist`  |  This function loops through a directory and builds friend/follower network in an edgelist format. |
| `get_friends_for_id`  |  Gets ALL friend IDS for a given user Adapted from Mike K's code. |
| `get_hash`  |  Returns list of hashtags in a tweet.  If no hashtags, it returns an empty list. |
| `get_mention` |  Returns list of mentions in a tweet.  If no hashtags, it returns an empty list. |
| `get_reply_conversation`  |  Recursively extracts replies and replies to replies in order to pull all replies that are connected to a given status_id(s) |
| `get_sensitivity` | Returns sensitivity  |
| `get_urls`  |  Returns list of expanded urls in a tweet.  If no urls, returns an empty list. |
| `get_user_map`  |  Function provided by Tom Magelinski Creates mapping from old screen names to new screen names based on retweet data |
| `parse_only_ids`  |  This parses 'tweet' json to a pandas dataFrame, but only gets the text, user id, tweet id, and language settings. 'name' should be either 'id_str' or 'screen_name'. |
| `parse_only_text`  |  This parses 'tweet' json to a pandas dataFrame, but only gets the text, user id, tweet id, and language settings. |
| `parse_twitter_json`  |  This parses 'tweet' json to a pandas dataFrame. |
| `plot_time`  |  This is a quick plot function that will create the data density for tweets in a single file or list of tiles. Prints matplotlib to screen |
| `rehydrate`  |  A wrapper method around tweepy.API.statuses_lookup that handles the batch lookup of Tweet IDs. |
