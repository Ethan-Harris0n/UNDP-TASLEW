__version__ = 'dev'


'''Include twitter manipulation functions here and move ngram extraction to dedicated script'''
# ### N-Gram Extraction
# #### Get top n-grams function

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer

# def get_ngrams(data, text,  n=5, ngram=2, varname='bigrams'):
#     '''
#     Main function to extract ngrams given a series of tweets via the two helpers belwo
#     ARGS:
#         - data = the twitter dataframe
#         - text = column with text for keywords to be extracted
#         - n for number of keywords
#         - ngram for range of ngrams to extract - currently only supports full factorial (e.g. 2 = 1 an 2 ngrams)
#     '''  
#     pop_bigrams = extract_ngrams(text, n, ngram)
#     data[varname] = data.text.apply(lambda x:(add_ngrams(x,pop_bigrams)))
#     ngramdf = data.explode(varname)
#     ngramdf = ngramdf[f'{varname}']
#     return ngramdf



# def extract_ngrams(corpus, n=5, ngram=2):
#     '''
#     Helper function to vectorize and extracts ngrams based on provided corpus 
#     ARGS:
#         - corpus is the text of interest
#         - n is the number of keywords to extract
#         - ngram is the range of ngrams to extract - this is set to full-factorial (e.g. 2 by 2) ngrams) currently but can change later on. 
    
#     '''
#     vec1 = CountVectorizer(ngram_range=(ngram,ngram), 
#            max_features=20000).fit(corpus)
#     ##TIFID RESULTS WERE POOR (Would need to likely filter by part of speech first)
#     # vec1 = CountVectorizer(ngram_range=(ngram,ngram), 
#     #         max_features=2000).fit(corpus)    
#     bag_of_words = vec1.transform(corpus)
#     sum_words = bag_of_words.sum(axis=0) 
#     print(sum_words)
#     words_freq = [(word, sum_words[0, idx]) for word, idx in     
#                   vec1.vocabulary_.items()]
#     words_freq =sorted(words_freq, key = lambda x: x[1], 
#                 reverse=True)
#     ngrams = pd.DataFrame(words_freq[:n])
#     return ngrams
# # Check if words_freq is same as df to list

# def add_ngrams(x, ngrams):
#     '''
#     Helper function for adding ngrams to associated text in a lambda call  /
#     (likely inefficient but not worth doing it via another process at this stage)
#     '''
#     lsofgrams = ngrams[0].to_list()
#     ls = []
#     for g in lsofgrams:
#         if g in x:
#             ls.append(g)
#     return ls