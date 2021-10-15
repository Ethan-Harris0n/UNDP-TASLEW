# Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
'''
A class for implementing various sentiment analysis frameworks:
- Vader is a rule-based sentiment classifier that is desgined for use with Social Media specifiically. 
    It can be somewhat limited however given it only accounts for a specific word and ignores the context in which it was used - this can cause weird results at times.
- Flair is


'''
class SentimentModel(object):
    def __init__(self, df, text_col, model):
            self.df = df
            self.text_col = text_col
            self.model = model
            

    def fit_model(self, **kwargs):
        model_list = ['vader']
        # filt = self.df[self.text_col]
        # if self.model == 'vadar':
        #     analyzer = SentimentIntensityAnalyzer()
        #     self.df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in filt]
        #     self.df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in filt]
        #     self.df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in filt]
        #     self.df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in filt]
        if self.model == 'vadar':
            analyzer = SentimentIntensityAnalyzer()
            self.df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in (self.df[self.text_col])]
            self.df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in (self.df[self.text_col])]
            self.df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in (self.df[self.text_col])]
            self.df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in (self.df[self.text_col])]
            return self.df


    
        else:
            return f"The model parameter you listed is not a valid input please select one of the following model paramters: {model_list}"

    def test(self, **kwargs):
        return filt
