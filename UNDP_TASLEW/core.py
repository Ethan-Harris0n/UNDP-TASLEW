# Libraries
import pandas as pd
import pandas_flavor as pf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import nltk


#######
'''
- core.py includes some brief data methods and basic text extraction methods
- all of these are built using the pandas_flavor library to integrate with pandas as addtional methods attached to the pandas dataframe class definition
- this is done by including @pf.register_dataframe_method prior to any method / function definition
'''

######################################
### Helper Functions ####
def extract_ngrams(corpus, n=5, ngram=2):
    '''
    Helper function to vectorize and extracts ngrams based on provided corpus 
    ARGS:
        - corpus is the text of interest
        - n is the number of keywords to extract
        - ngram is the range of ngrams to extract - this is set to full-factorial (e.g. 2 by 2) ngrams) currently but can change later on. 
    
    '''
    vec1 = CountVectorizer(ngram_range=(ngram,ngram), 
           max_features=20000).fit(corpus)
    ##TIFID RESULTS WERE POOR (Would need to likely filter by part of speech first)
    # vec1 = CountVectorizer(ngram_range=(ngram,ngram), 
    #         max_features=2000).fit(corpus)    
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    print(sum_words)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    ngrams = pd.DataFrame(words_freq[:n])
    return ngrams
# Check if words_freq is same as df to list
def add_ngrams(x, ngrams):
    '''
    Inefficient - OPTIMIZE!!!
    Helper function for adding ngrams to associated text in a lambda call  /
    (likely inefficient but not worth doing it via another process at this stage)
    '''
    lsofgrams = ngrams[0].to_list()
    ls = []
    for g in lsofgrams:
        if g in x:
            ls.append(g)
    return ls


# Function to sort scores pulled from a stack overflow post
def sort_scores(matrix):
    tuples = zip(matrix.col, matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


# Function to get feature names and TFIDF score of top n items
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

#####################################################################
#####################################################################
#### Core Methods ####
'''
Many of these use pandas_flavor as a decortator to extend pandas - i considered making a custom text dataframe class or creating a new set of class methods for each core function.
However, I believe this to be the most simplistic solution and therefore the best from a replicabilibility standpoint
'''
#%%
@pf.register_dataframe_method
def get_text_df(df, col):
    """reduce dataframe to index and column to be analyzed"""
    return df[[col]]

#%%
@pf.register_dataframe_method
def get_fulldf(df, old_df):
        '''
        returns merge of Text_DF with original dataframe
        '''
        remerge = df.merge(old_df)
        return remerge

#%%
@pf.register_dataframe_method
def get_tfidf(df, old_df):
        '''
        returns a term-frequency-i
        '''
        remerge = df.merge(old_df)
        return remerge
#%%
@pf.register_dataframe_method
def get_ngrams(df,  n=5, ngram=2, varname='ngrams'):
    '''
    Main function to extract ngrams given a series of tweets via the two helpers belwo
    ARGS:
        - df = the twitter dataframe
        - text = column with text for keywords to be extracted
        - n for number of keywords
        - ngram for range of ngrams to extract - currently only supports full factorial (e.g. 2 = 1 an 2 ngrams)
    '''  
    pop_bigrams = extract_ngrams(df.iloc[:,0], n, ngram)
    df[varname] = df.iloc[:,0].apply(lambda x:(add_ngrams(x,pop_bigrams))) # inefficient - need to convert to list comprehension
    ngramdf = df.explode(varname)
    ngramdf = ngramdf[f'{varname}']
    return ngramdf


#%%
'''
Next steps:
- Create function so custom tifid transformer can be passed in
- Include get keywords than allow type selection (e.g. bert versus TFIDF)
'''
stop_words = set(stopwords.words("english"))
@pf.register_dataframe_method
def get_keywords_tfidf(df, nwords, varname='keywords', wordtype=True):
    # Gen Corpus
    corpus = df.iloc[:,0].tolist()
    # NLTK POS tagger breaks when I try to pass it the twitter id as index - am not sure why this is - pretty confusing aas it should have been easy fix
    data = df.copy().reset_index()
    # Initialize TFIDF Model
    tfidfmodel = TfidfTransformer(smooth_idf=True,use_idf=True) # number of parameters to play around with here
    # Vectorize data
    cv=CountVectorizer(max_df=0.95,stop_words=stop_words) # and...here
    word_count_vector=cv.fit_transform(corpus)
    # Fit TFIDF Model
    tfidfmodel.fit(word_count_vector)
    tf_idf_vector=tfidfmodel.transform(cv.transform(corpus))
    feature_names = cv.get_feature_names()
    results=[]
    for i in range(tf_idf_vector.shape[0]):
        #Current vector
        curr_vector=tf_idf_vector[i]
        #sort the tf-idf vector by descending order of scores
        sorted_items=sort_scores(curr_vector.tocoo())
        #extract top n
        keywords=extract_topn_from_vector(feature_names,sorted_items,nwords)
        results.append(keywords)
    keywords=pd.DataFrame(zip(corpus,results),columns=['corpus_text','keywords'])
    y = data.join(keywords)
    # Long format for CRD visualization
    keywords_df = y.explode('keywords')
    keywords_df = keywords_df[['id','keywords']]
    # return keywords_df
    if wordtype==True:
        keywords_df['POS'] = nltk.pos_tag(keywords_df.keywords)
        keywords_df['keywords'], keywords_df['POS'] = zip(*keywords_df['POS'])
        keywords_df.set_index('id',inplace=True)
        return keywords_df
    else:
        keywords_df.set_index('id',inplace=True)
        return keywords_df
#%%
### Under Development - Vectorizer Class ###
class Vectorize_Dataframe(object):
    '''
    Takes a df and collumn of text as input and converts it to either a count vectorized or tfidf matrix
    '''
    def __init__(self, df, text_column, vectorizer='tfidf', **vectorizer_kwargs):
        vecs = ['tfidf', 'cv']
        self.corpus = df
        self.text_column = text_column
        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(decode_error="ignore", **vectorizer_kwargs)
            self.tfidf = self.vectorizer.fit_transform(df[text_column])
        if vectorizer == 'countvectorizer':
            self.vectorizer = CountVectorizer(**vectorizer_kwargs)
            self.cv = self.vectorizer.fit_transform(df[text_column])
        else:
            raise Exception (
                f'You inputted an invalid option for the vectorizer parameter please specify one of the following: {vecs}')
# %%
    def find_related_keywords(self, keyword, n=25):
        """
        Given a particular keyword, looks for related terms in the corpus using mutual information.
        :param keyword: The keyword to use
        :type keyword: str
        :param n: Number of related terms to return
        :type n: int
        :return: Terms associated with the keyword
        :rtype: list
        Usage::
            >>> tdf.find_related_keywords("war")[:2]
            ['war', 'peace']
            >>> tdf.find_related_keywords("economy")[:2]
            ['economy', 'expenditures']
        """

        self.corpus["temp"] = (
            self.corpus[self.text_column]
            .str.contains(r"\b{}\b".format(keyword), re.IGNORECASE)
            .astype(int)
        )
        mi = self.mutual_info("temp")
        del self.corpus["temp"]

        return list(mi[mi["MI1"] > 0].sort_values("MI1", ascending=False)[:n].index)

#%%
    def make_word_cooccurrence_matrix(
            self, normalize=False, min_frequency=10, max_frequency=0.5
        ):

            """
            Use to produce word co-occurrence matrices. Based on a helpful StackOverflow post:
            https://stackoverflow.com/questions/35562789/how-do-i-calculate-a-word-word-co-occurrence-matrix-with-sklearn
            :param normalize: If True, will be normalized
            :type normalize: bool
            :param min_frequency: The minimum document frequency required for a term to be included
            :type min_frequency: int
            :param max_frequency: The maximum proportion of documents containing a term allowed to include the term
            :type max_frequency: int
            :return: A matrix of (terms x terms) whose values indicate the number of documents in which two terms co-occurred
            Usage::
                >>> wcm = tdf.make_word_cooccurrence_matrix(min_frequency=25, normalize=True)
                # Find the top cooccurring pair of words
                >>> wcm.stack().index[np.argmax(wcm.values)]
                ('protection', 'policy')
            """

            text = self.corpus[self.text_column]
            cv = CountVectorizer(
                ngram_range=(1, 1),
                stop_words="english",
                min_df=min_frequency,
                max_df=max_frequency,
            )
            mat = cv.fit_transform(text)
            mat[
                mat > 0
            ] = (
                1
            )  # this makes sure that we're counting number of documents words have in common \
            # and not weighting by the frequency of one of the words in a single document, which can lead to spurious links
            names = cv.get_feature_names()
            mat = mat.T * mat  # compute the document-document matrix
            if normalize:
                diag = sp.diags(1.0 / mat.diagonal())
                mat = diag * mat
            mat.setdiag(0)
            matrix = pd.DataFrame(data=mat.todense(), columns=names, index=names)

            return matrix

# %%
