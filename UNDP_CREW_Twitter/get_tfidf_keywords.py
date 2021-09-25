#### Function to extract TFIDF values
#### Function to extract TFIDF values
## SKlearn - Needed for vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#### Function to extract TFIDF values
def extract_tfidf_keywords(data, nwords, varname='keywords', wordtype=True):
    df = data.copy().reset_index()
    # Gen Corpus
    corpus = data.text_clean.tolist()
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
    y = df.join(keywords)
    y.set_index(df.id, inplace=True)
    # Long format for CRD visualization
    keywords_df = y.explode('keywords')
    keywords_df = keywords_df[['keywords']]
    if wordtype==True:
        keywords_df['POS'] = nltk.pos_tag(keywords_df.keywords)
        keywords_df['keywords'], keywords_df['POS'] = zip(*keywords_df['POS'])
        return keywords_df
    else:
        return keywords_df



### TF-IDF Core Functions
#### Function to sort scores pulled from a stack overflow post
def sort_scores(matrix):
    tuples = zip(matrix.col, matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

#### Function to get feature names and TFIDF score of top n items

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
