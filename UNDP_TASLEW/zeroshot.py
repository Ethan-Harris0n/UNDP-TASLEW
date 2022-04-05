from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

class zeroshot(object):
    def __init__(self, df, text_col, labels, gpu=-1, multilabel=False, n_jobs=-1):
            self.df = df
            self.text_col = text_col
            self.labels = labels
            self.gpu=gpu
            self.multilabel = multilabel
            self.n_jobs = n_jobs

    def add_labels(self, new_labels):
        self.labels.append(new_labels)

    def fit_model(self):
        classifier = pipeline("zero-shot-classification", device=self.gpu)
        # data['label_probs'] = data.text_clean.apply(lambda x:classifier(x, classes, multilabel=multilabel))
        self.df['label_probs'] = [classifier(x, self.labels,  multilabel=self.multilabel)['label_probs'] for x in (self.df[self.text_col])]
        if self.multilabel==True:
            self.df.explode('label_probs', inplace=True)
            return self.df
        else:
            return self.df