# Generic Libraries
from __future__ import division
import pandas as pd
import pandas_flavor as pf
import math
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import copy

## Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

## Translation
from translate import Translator

## Nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tag import perceptron

#######
'''
- core.py includes some brief data methods and basic text extraction methods
- Some of these are built using the pandas_flavor library to integrate with pandas as addtional methods
- this is done by including @pf.register_dataframe_method prior to any method / function definition
- There is also a basic vectorizer dataframe class that we should continue to add additional methods to over time
'''

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
        remerge = df.merge(old_df, how='right', **kwargs)
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
        df[f'text_{lang}']= [translator.translate(x) for x in (df[df_col])]
        return df

#%%
@pf.register_dataframe_method
def clean_text(df, df_col, **preprocess_kwargs):
    '''
    preprocess for text
    preprocess_kwargs:
    - stop_words: list of stop words
    - remove_tags=True: remove hashtags
    - remove_special_characters=True: remove special characters
    - remove_digits=True: removes digits
    - stem=True: (default equals "False") Stems words via NLTK's porterstemmer
    - lemmatize=True: lemmatizes words via NLTK's wordnet lemmatizer
    See helper functions at bottom of script for more info
    '''
    df_col = df[df_col].astype(str)
#     df['text_clean'] = [pre_process(x,**preprocess_kwargs)['text_clean'] for x in (df_col)] 
    df['text_clean'] = df_col.apply(lambda x:pre_process(x,**preprocess_kwargs))
    return df

#%%
@pf.register_dataframe_method
def make_word_cooccurrence_matrix(
            df, df_col, normalize=False, min_frequency=10, max_frequency=0.5
        ):

            """
            Should probably convert this under vectorize class but given strict ngram requirements keeping here for now

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

            text = df[df.col]
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
- Include get keywords than allow type selection (e.g. bert versus TFIDF)
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

        self.df["temp"] = (
            self.df[self.df.columns.values[0]]
            .str.contains(r"\b{}\b".format(keyword), re.IGNORECASE)
            .astype(int)
        )
        mi = self.mutual_info("temp")
        del self.df["temp"]

        return list(mi[mi["MI1"] > 0].sort_values("MI1", ascending=False)[:n].index)

    def mutual_info(
            self, outcome_col, weight_col=None, sample_size=None, l=0, normalize=True
        ):

            """
            A wrapper around of the compute mutual information helper used in find related keywords func above`
            :param outcome_col: The name of the column with the binary outcome variable
            :type outcome_col: str
            :param weight_col: (Optional) Name of the column to use in weighting
            :type weight_col: str
            :param sample_size: (Optional) If provided, a random sample of this size will be used instead of the full \
            :py:class:`pandas.DataFrame`
            :type sample_size: int
            :param l: An optional Laplace smoothing parameter
            :type l: float
            :param normalize: Toggle normalization on or off (to control for feature prevalence), on by default
            :type normalize: bool
            :return: A :py:class:`pandas.DataFrame` of ngrams and various metrics about them, including mutual information
            """

            keep_columns = [self.text_column, outcome_col]
            if weight_col:
                keep_columns.append(weight_col)
            df = copy.deepcopy(self.corpus[keep_columns])
            if sample_size:
                df = df.sample(n=sample_size).reset_index()
            if weight_col:
                df = df.dropna().reset_index()
            else:
                df = df.dropna(subset=[self.text_column, outcome_col]).reset_index()
            y = df[outcome_col]
            x = self.vectorizer.transform(df[self.text_column])
            weights = None
            if weight_col:
                weights = df[weight_col]

            return compute_mutual_info(
                y,
                x,
                weights=weights,
                col_names=self.vectorizer.get_feature_names(),
                l=l,
                normalize=normalize,
            )

##### Helpers #####

def pre_process(text, remove_tags=True, remove_special_characters=True, 
                remove_digits=True, lemmatize=True, stem=False):
                
                # Stop words
                stop_words = set(stopwords.words("english"))
                # Raise value error if try to lemmatize and stem
                if lemmatize==True & stem==True:
                        raise ValueError('Please set stem or lemmatize arguement to "False"')
                
                # lowercase
                text=text.lower()
                
                if remove_tags==True: # for keyword extraction
                        #remove tags
                        text = re.sub("#+","", text)

                if remove_special_characters==True:
                        # remove special characters
                        text=re.sub("([^\w#])+"," ",text)
                
                if remove_digits==True:
                        # remove digits
                        text=re.sub("(\\d)+"," ",text)

                
                if stem==True:
                        #convert to list
                        text = text.split()
                        #stem
                        ps=PorterStemmer()
                        text = [ps.stem(word) for word in text if not word in  
                        stop_words] 
                        text = " ".join(text)
                
                if lemmatize==True:
                        #convert to list
                        text = text.split()
                        #Lemmatisation
                        lem = WordNetLemmatizer()
                        text = [lem.lemmatize(word) for word in text if not word in  
                                stop_words] 
                        text = " ".join(text)
                return text


##### Keyword Helpers #################################################
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


### Helpers pulled from pewanalytics !!! - https://github.com/pewresearch/pewtils/blob/main/pewtils/__init__.py
def is_not_null(val, empty_lists_are_null=False, custom_nulls=None):

    """
    Checks whether the value is null, using a variety of potential string values, etc. The following values are always
    considered null: ``numpy.nan, None, "None", "nan", "", " ", "NaN", "none", "n/a", "NONE", "N/A"``
    :param val: The value to check
    :param empty_lists_are_null: Whether or not an empty list or :py:class:`pandas.DataFrame` should be considered \
    null (default=False)
    :type empty_lists_are_null: bool
    :param custom_nulls: an optional list of additional values to consider as null
    :type custom_nulls: list
    :return: True if the value is not null
    :rtype: bool
    Usage::
        from core.py import is_not_null
        >>> text = "Hello"
        >>> is_not_null(text)
        True
    """

    null_values = [None, "None", "nan", "", " ", "NaN", "none", "n/a", "NONE", "N/A"]
    if custom_nulls:
        null_values.extend(custom_nulls)
    if type(val) == list:
        if empty_lists_are_null and val == []:
            return False
        else:
            return True
    elif isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
        if empty_lists_are_null and len(val) == 0:
            return False
        else:
            return True
    else:
        try:
            try:
                good = val not in null_values
                if good:
                    try:
                        try:
                            good = not pd.isnull(val)
                        except IndexError:
                            good = True
                    except AttributeError:
                        good = True
                return good
            except ValueError:
                return val.any()
        except TypeError:
            return not isinstance(val, None)



def scale_range(old_val, old_min, old_max, new_min, new_max):

    """
    Scales a value from one range to another.  Useful for comparing values from different scales, for example.
    :param old_val: The value to convert
    :type old_val: int or float
    :param old_min: The minimum of the old range
    :type old_min: int or float
    :param old_max: The maximum of the old range
    :type old_max: int or float
    :param new_min: The minimum of the new range
    :type new_min: int or float
    :param new_max: The maximum of the new range
    :type new_max: int or float
    :return: Value equivalent from the new scale
    :rtype: float
    """

    return (
        ((float(old_val) - float(old_min)) * (float(new_max) - float(new_min)))
        / (float(old_max) - float(old_min))
    ) + float(new_min)

def compute_mutual_info(y, x, weights=None, col_names=None, l=0, normalize=True):

    """
    Computes pointwise mutual information for a set of observations partitioned into two groups.
    :param y: An array or, preferably, a :py:class:`pandas.Series`
    :param x: A matrix, :py:class:`pandas.DataFrame`, or preferably a :py:class:`scipy.sparse.csr_matrix`
    :param weights: (Optional) An array of weights corresponding to each observation
    :param col_names: The feature names associated with the columns in matrix 'x'
    :type col_names: list
    :param l: An optional Laplace smoothing parameter
    :type l: int or float
    :param normalize: Toggle normalization on or off (to control for feature prevalance), on by default
    :type normalize: bool
    :return: A :py:class:`pandas.DataFrame` of features with a variety of computed metrics including mutual information.
    The function expects ``y`` to correspond to a list or series of values indicating which partition an observation \
    belongs to. ``y`` must be a binary flag. ``x`` is a set of features (either a :py:class:`pandas.DataFrame` or \
    sparse matrix) where the rows correspond to observations and the columns represent the presence of features (you \
    can technically run this using non-binary features but the results will not be as readily interpretable.) The \
    function returns a :py:class:`pandas.DataFrame` of metrics computed for each feature, including the following \
    columns:
    - ``MI1``: The feature's mutual information for the positive class
    - ``MI0``: The feature's mutual information for the negative class
    - ``total``: The total number of times a feature appeared
    - ``total_pos_with_term``: The total number of times a feature appeared in positive cases
    - ``total_neg_with_term``: The total number of times a feature appeared in negative cases
    - ``total_pos_neg_with_term_diff``: The raw difference in the number of times a feature appeared in positive cases \
    relative to negative cases
    - ``pct_pos_with_term``: The proportion of positive cases that had the feature
    - ``pct_neg_with_term``: The proportion of negative cases that had the feature
    - ``pct_pos_neg_with_term_ratio``: A likelihood ratio indicating the degree to which a positive case was more likely \
    to have the feature than a negative case
    - ``pct_term_pos``: Of the cases that had a feature, the proportion that were in the positive class
    - ``pct_term_neg``: Of the cases that had a feature, the proportion that were in the negative class
    - ``pct_term_pos_neg_diff``: The percentage point difference between the proportion of cases with the feature that \
    were positive vs. negative
    - ``pct_term_pos_neg_ratio``: A likelihood ratio indicating the degree to which a feature was more likely to appear \
    in a positive case relative to a negative one (may not be meaningful when classes are imbalanced)
    .. note:: Note that ``pct_term_pos`` and ``pct_term_neg`` may not be directly comparable if classes are imbalanced, \
        and in such cases a ``pct_term_pos_neg_diff`` above zero or ``pct_term_pos_neg_ratio`` above 1 may not indicate a \
        true association with the positive class if positive cases outnumber negative ones.
    .. note:: Mutual information can be a difficult metric to explain to others. We've found that the \
        ``pct_pos_neg_with_term_ratio`` can serve as a more interpretable alternative method for identifying \
        meaningful differences between groups.
    """

    if is_not_null(weights):
        weights = weights.fillna(0)
        y0 = sum(weights[y == 0])
        y1 = sum(weights[y == 1])
        total = sum(weights)
    else:
        y0 = len(y[y == 0])
        y1 = len(y[y == 1])
        total = y1 + y0

    if type(x).__name__ == "csr_matrix":

        if is_not_null(weights):
            x = x.transpose().multiply(csr_matrix(weights)).transpose()
        x1 = pd.Series(x.sum(axis=0).tolist()[0])
        x0 = total - x1
        x1y0 = pd.Series(
            x[np.ravel(np.array(y[y == 0].index)), :].sum(axis=0).tolist()[0]
        )
        x1y1 = pd.Series(
            x[np.ravel(np.array(y[y == 1].index)), :].sum(axis=0).tolist()[0]
        )

    else:

        if type(x).__name__ != "DataFrame":
            x = pd.DataFrame(x, columns=col_names)

        if is_not_null(weights):
            x = x.multiply(weights, axis="index")
            x1 = x.multiply(weights, axis="index").sum()
            x0 = ((x * -1) + 1).multiply(weights, axis="index").sum()
        else:
            x1 = x.sum()
            x0 = ((x * -1) + 1).sum()
        x1y0 = x[y == 0].sum()
        x1y1 = x[y == 1].sum()

    px1y0 = x1y0 / total
    px1y1 = x1y1 / total
    px0y0 = (y0 - x1y0) / total
    px0y1 = (y1 - x1y1) / total

    px1 = x1 / total
    px0 = x0 / total
    py1 = float(y1) / float(total)
    py0 = float(y0) / float(total)

    MI1 = (px1y1 / (px1 * py1) + l).map(lambda v: math.log(v, 2) if v > 0 else 0)
    if normalize:
        MI1 = MI1 / (-1 * px1y1.map(lambda v: math.log(v, 2) if v > 0 else 0))

    MI0 = (px1y0 / (px1 * py0) + l).map(lambda v: math.log(v, 2) if v > 0 else 0)
    if normalize:
        MI0 = MI0 / (-1 * px1y0.map(lambda v: math.log(v, 2) if v > 0 else 0))

    df = pd.DataFrame()

    df["MI1"] = MI1
    df["MI0"] = MI0

    df["total"] = x1
    df["total_pos_with_term"] = x1y1  # total_pos_mention
    df["total_neg_with_term"] = x1y0  # total_neg_mention
    df["total_pos_neg_with_term_diff"] = (
        df["total_pos_with_term"] - df["total_neg_with_term"]
    )
    df["pct_with_term"] = x1 / (x1 + x0)
    df["pct_pos_with_term"] = x1y1 / y1  # pct_pos_mention
    df["pct_neg_with_term"] = x1y0 / y0  # pct_neg_mention
    df["pct_pos_neg_with_term_diff"] = (
        df["pct_pos_with_term"] - df["pct_neg_with_term"]
    )  # pct_pos_neg_mention_diff
    df["pct_pos_neg_with_term_ratio"] = df["pct_pos_with_term"] / (
        df["pct_neg_with_term"]
    )  # pct_pos_neg_mention_ratio

    df["pct_term_pos"] = x1y1 / x1  # pct_mention_pos
    df["pct_term_neg"] = x1y0 / x1  # pct_mention_neg
    df["pct_term_pos_neg_diff"] = (
        df["pct_term_pos"] - df["pct_term_neg"]
    )  # pct_mention_pos_neg_diff
    df["pct_term_pos_neg_ratio"] = df["pct_term_pos"] / df["pct_term_neg"]

    if col_names:
        df.index = col_names

    return df