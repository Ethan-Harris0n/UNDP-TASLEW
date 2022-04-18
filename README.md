## TASLEW: Text Analysis and Social Listening for Early Warning

A package built by the UNDP's Crisis Risk and Early Warning team to extract, transform, and analyze text data for Early Warning purposes.


## The following repos provided information and/or source code for many of the functions contained in this package
- https://github.com/MichaelKim0407/tutorial-pip-package - for tutorial on building a python package
- https://github.com/heatherbaier/ - for highlighting said tutorial above
- https://github.com/pewresearch/pewanalytics - several of their co-iccurences / mutual information funcs (modified versions sourced from stack overflow) have been "canabalized" and wrapped under the vectorizer class - chose to do this instead of importing their package due to a number of install issues related to their build 
- https://github.com/dmbeskow/twitter_col - wrapped a number of his excellent functions into a twitter module for ad-hoc data cleaning during data-extraction tasks on databricks

### Summary of methods

An itemized list of methods can be found in the tables below. Typically these functions take a dataframe and column of text as inputs and output a dataframe constituting of the original dataframe's index and outputted column(s) produced by the method that was called.


#### Generic Text DataFrame Methods (core.py)
These are functions added via a pandas decorator. They thus can be called on any pandas dataframe e.g. `df.translate_text('column 1', to_lang='eng)` will automatically translate the text in column 1 of the dataframe to english.
.
| Function | Description |
|----------|-------------|
| `get_text_df`  | Preps a dataframe for analysis by reducing it to only its unique ID and text to be analyzed |

| `get_fulldf`  | Simple wrapper for re-merging that I found myself using. Merges current text dataframe to original dataframe based on a common ID |

| `translate_text`  |  Quick and dirty function for translating text using google translate's API|

| `clean_text`  |  Function with several arguments (see core.py for more details) to clean text for text analysis|

| `get_ngras`  |  Function with several arguments (see core.py for more details) to extract ngrams (phrases) from a text dataframe. Returns a long df with ngrams corresponding to the original index|

| `find_related_keywords`  | Takes keywords or a list of keywords as input and produces a list of related keywords - useful for unsupervised lexicon building and expansion|

| `make_word_cooccurrence_matrix` | Generates a co-iccurence matrix for a inputted document - useful for unsupervised lexicon building and expansion.|

##### The Vectorize Dataframe class
The function `Vectorize_Dataframe` takes a dataframe and column of text as input and converts it to either a count vectorized or tf-idf matrix. Can feed additional keyword arguments from Sklearn's CountVectorizer or TfidfVectorizer to the constructor.

Initializing this object than allows the user to call the following functions:
| Function | Description |
|----------|-------------|
| `get_keywords` | Extracts keywords from text based on tf-idf. There is also an additional keyword arguement that allows the user to filter for parts of speech (e.g. nouns). |

| `find_related_keywords` | locates similar words in vector-space based on a user-specified keyword. Useful for expanding lexicons. |


#### Databricks Functions
 These are a collection of helper functions used frequently on databricks (primarily for social-meda analysis). They include:

| Function | Description |
|----------|-------------|
| `get_last_id(table)`  | Gets most recent tweet ID from user-specified delta table. Used automate scheduled API calls for data updation |

| `get_last_timestamp(table)` | Extracts timestamp from most recent facebook post from specified delta table. Used to automate scheduled API calls for data updation  |

| `get_user_id` | Pulls the id of a twitter-user based on inputted twitter handle. Used primarily for data extraction tasks. |

| `bearer_oauth` | Used for twitter authentication in API calls |

| `connect_to_endpoint` | Used for twitter data extraction tasks |

| `pull` | helper to call a twitter endpoint |


##### Topic.py
Generate topic-analysis of text using a variety of methods (Latent-Derilicht Allocation, Correlation Explanation, Bert, etc.) Cannibalized from Pew Research's repo - Pew Analytics|

##### Zeroshot.py
This is really just one method but added it as a module to improve ease in application. Zero Shot is an implementation of Hugging Face's Zero Shot Classifcation framework originating from the following blog post: https://joeddav.github.io/blog/2020/05/29/ZSL.html
##### Sentiment.py
This module enables one to calculate text entiment via a number of frameworks including Facebook's Vader, Flair,etc. 

##### Hate.py
Module devoted to Hate-Speech Classification. Currently this only includes Detoxify from Unity-AI's framework.

#### Twitter Manipulation Methods (tweets.py)
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
