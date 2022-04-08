from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline


from transformers import pipeline

class zeroshot(object):
    def __init__(self, text_col, labels, **pipeline_kwargs):
            # self.df = df
            self.df = pd.DataFrame(text_col).set_index(text_col.index)
            self.text_col = text_col
            self.labels = labels
            self.classifier = pipeline("zero-shot-classification", **pipeline_kwargs)

    def add_labels(self, new_labels):
        self.labels.append(new_labels)

    def fit_model(self):
        self.df['output'] = self.text_col.apply(lambda x:self.classifier(x, self.labels))
        self.df['labels'] = self.df['output'].apply(lambda x: x.get('labels'))
        self.df['scores'] = self.df['output'].apply(lambda x: x.get('scores'))
        self.df.drop('output', axis=1)
        self.df = self.df.apply(lambda x: x.explode() if x.name in ['labels','scores'] else x) # there's a more efficient way to do this / in the future probably should set up an if / else with multilabel user-specified argument
        return self.df