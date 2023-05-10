def get_result(series):
    import pandas as pd
    import pickle
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import BaseEstimator, TransformerMixin
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    import pymorphy2

    def data_prep(series):
        series = series.str.replace('_x000D_', '')
        series = series.str.replace('[^\w\s]', ' ', regex=True)
        series = series.str.replace('\s+', ' ', regex=True)
        series = series.str.lower()

        stop_words = set(stopwords.words('russian'))
        series = series.apply(
            lambda review: ' '.join([word for word in review.split() if word not in stop_words]))

        morph = pymorphy2.MorphAnalyzer()
        def lemmatizer(sentence):
            words = sentence.split()
            normal_words = [
                morph.parse(word)[0].normal_form
                for word in words
            ]
            return ' '.join(normal_words)
        series = series.apply(lemmatizer)

        return series
    
    preproped_reviews = data_prep(series)

    with open('model_pkl', 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(preproped_reviews)

    predictions = pd.Series(
        predictions, index=series.index, name='class_predicted').map(
            {0: 'Доставка', 1: 'Товар', 2: 'Магазин'})

    return predictions