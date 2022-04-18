from detoxify import Detoxify

class HateSpeech_Model(object):
    '''
    A class for innitializing a number of hate-speech models
    and applying them in a number of languages
    '''
    def __init__(self, df, text_col_string, model):
            self.df = df
            self.text_col_string = text_col_string
            self.model = model
            

    def fit_model(self, **kwargs):
        model_list = ['detoxify']
        if self.model == 'detoxify':
            model = Detoxify(**kwargs)
            self.df['detoxify_output'] = [model.predict(x)['detoxify_output'] for x in (self.df[self.text_col_string])]
            # self.df['detoxify_output'] = [model.predict(x) for x in (self.df[self.text_col])]
            return self.df