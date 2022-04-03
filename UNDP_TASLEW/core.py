# Libraries
import pandas as pd
import pandas_flavor as pf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from translate import Translator
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

def extract_tfidf_keywords(matrix, matrix_names, nwords):
    results=[]
    for i in range(matrix.shape[0]):
        #Current vector
        curr_vector=matrix[i]
        #sort the tf-idf vector by descending order of scores
        sorted_items=sort_scores(curr_vector.tocoo())
        #extract top n
        keywords=extract_topn_from_vector(matrix_names,sorted_items,nwords)
        results.append(keywords)
    return results

#####################################################################
#####################################################################
#### Core Methods ####
'''
Many of these use pandas_flavor as a decortator to extend pandas - i considered making a custom text dataframe class or creating a new set of class methods for each core function.
However, I believe this is the most simplistic solution and thus the most flexible.
'''
#%%
@pf.register_dataframe_method
def get_text_df(df, col):
    """reduce dataframe to index and column to be analyzed"""
    return df[[col]]

#%%
@pf.register_dataframe_method
def get_fulldf(df, old_df, **kwargs):
        '''
        returns merge of Text_DF with original dataframe
        '''
        remerge = df.merge(old_df, **kwargs)
        return remerge

#%%
@pf.register_dataframe_method
def translate_text(df, df_col, lang="en"):
        '''
        Quick and dirty function for translating text
        should consider adding a cleaining step to split words
        should explore alternatives (are any of the hugging face models worth pursuing)
        '''
        translator= Translator(to_lang=lang)
        df[f'text_{lang}']= [translator.translate(x) for x in (df['df_col'])]
        return df



#%%
##### Keywords Module #################################################
### Helper Functions ####
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

# Method
#### Function to extract TFIDF values
def extract_keywords(matrix, matrix_names, nwords):
    results=[]
    for i in range(matrix.shape[0]):
        #Current vector
        curr_vector=matrix[i]
        #sort the tf-idf vector by descending order of scores
        sorted_items=sort_scores(curr_vector.tocoo())
        #extract top n
        keywords=extract_topn_from_vector(matrix_names,sorted_items,nwords)
        results.append(keywords)
    return results

############################################################

#%%
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



'''
Next steps:
- Create function so custom tifid transformer can be passed in
- Include get keywords than allow type selection (e.g. bert versus TFIDF)
- Create translate function
- Add keywords extraction function
'''


### Under Development - Vectorizer Class ###
#%%
class Vectorize_Dataframe(object):
    '''
    Takes a df and collumn of text as input and converts it to either a count vectorized or tfidf matrix
    Can feed additional keyword arguments from Sklearn's CountVectorizer or TfidfVectorizer to the constructor
    '''
    def __init__(self, dataframe_column, vectorizer_type='tf-idf', **vectorizer_kwargs):
        vecs = ['tf-idf', 'cv']
        self.df = pd.DataFrame(dataframe_column).set_index(dataframe_column.index)
        self.corpus = dataframe_column.tolist()
        self.vectorizer_type = vectorizer_type
        if self.vectorizer_type == 'tf-idf':
            self.vectorizer = TfidfVectorizer(decode_error="ignore", **vectorizer_kwargs)
            self.tfidf = self.vectorizer.fit_transform(self.corpus)
        if self.vectorizer_type == 'cv':
            self.vectorizer = CountVectorizer(**vectorizer_kwargs)
            self.cv = self.vectorizer.fit_transform(self.corpus)
        if self.vectorizer_type not in vecs:
            raise ValueError('vectorizer must be one of {}'.format(vecs))

# %%
    def get_keywords(self, n=25, filter_N=True):
        '''
        extracts keywords based on tf-idf
        
        filter_N is a paramter for filtering out all keywords for nouns only.

        In the future this could be expanded to for more robust Parts of speech tagging and filtering 
                via a **kwargs argument
        '''        
        if self.vectorizer_type != 'tf-idf':
            raise ValueError('Keywords can only be extracted from tf-idf')
        else:
            # get features
            feature_names = self.vectorizer.get_feature_names_out()
            ## Extract keywords
            keywords = extract_keywords(self.tfidf, feature_names, n)
            ## bind dfs with original df and include corpus to confirm results via matching text
            kdf=pd.DataFrame(zip(self.corpus,keywords),columns=['corpus_text','keywords']).set_index(self.df.index)
            # y = self.df.join(kdf) # just to confirm matching indexes / correct merge
            # print(y)
            # Long format for CRD visualization
            keywords_df = kdf.explode('keywords')
            keywords_df = keywords_df[['keywords']]

            if filter_N==True:
                # Tag each word and create seperate column for part of speech
                keywords_df['keywords_wordtype'] = nltk.pos_tag(keywords_df.keywords)
                keywords_df['keywords'], keywords_df['POS'] = zip(*keywords_df['keywords_wordtype'])
                keywords_df_nouns = keywords_df[keywords_df['POS'].str.contains("N")]
                return keywords_df_nouns

            else:
                return keywords_df

# %%
    def find_related_keywords(self, keyword, n=25):
        """
        PEW
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
            PEW
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
