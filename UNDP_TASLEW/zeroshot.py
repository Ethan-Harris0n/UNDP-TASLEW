def zeroshot(data, classes, gpu=-1, multi_class=False, n_jobs=-1):
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", gpu)
    data['class_probs'] = data.text_clean.apply(lambda x:classifier(x, classes, multi_class=multi_class))
    if multi_class==True:
        data.explode('class_probs', inplace=True)
        return data
    else:
        return data