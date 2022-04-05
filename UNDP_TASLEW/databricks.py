###########
# Imports #
import pandas as pd
#### Helper Functions for Databricks / Social Media Analysis work #####

#### Twitter ####
# Get last tweet id of tweets collected
def get_last_id(table):
    last_id = spark.sql("SELECT MAX(id) as last_id FROM {}".format(table))
    return last_id.first()['last_id']

 # Pull id based on username
def get_user_id(username):
   userurl = pull("https://api.twitter.com/2/users/by/username/{}?user.fields=id".format(username))
   id = userurl['data']['id']
   return id


# Helper - auth
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    # r.headers["User-Agent"] = "v2ListTweetsLookupPython"
    return r

# Helper - connect to the endpoint
def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    #print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

# Helper - call endpoint
def pull(url):
    json_response = connect_to_endpoint(url)
    return json_response



#### Facebook ####

# For facebook API call - Function for getting last time stamp from most recent post in current database
def get_last_timestamp(table):
    last_ts = spark.sql("SELECT MAX(date) as last_timestamp FROM {}".format(table))
    return pd.to_datetime((last_ts.first()['last_timestamp']), format='%Y,%m,%d,%H,%M,%S')

