"""Historical notebook experiment; not independently validated."""
from src.data import load_data, split_data

def main():
    import pandas as pd
    final_df = load_data()
    train, test, val = split_data(final_df)
    BERT_train = train.copy()
    BERT_val = val.copy()
    BERT_test = test.copy()

    #Function to evaluate metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    def evaluate(val, val_predictions):
        val_precision, val_recall, val_f1, support = precision_recall_fscore_support(val['meta.genre'], val_predictions, average='macro', zero_division=True)
        val_accuracy = (val['meta.genre'] == val_predictions).mean()
        print(f'Validation accuracy: {val_accuracy:.3f}')
        print(f'Validation precision (macro-averaged): {val_precision:.3f}')
        print(f'Validation recall (macro-averaged): {val_recall:.3f}')
        print(f'Validation F1-score (macro-averaged): {val_f1:.3f}')




    from transformers import RobertaModel, RobertaTokenizer


    model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    import torch
    from transformers import RobertaTokenizer, RobertaModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, classification_report, confusion_matrix
    from transformers import pipeline
    from transformers import RobertaTokenizer, TFRobertaModel

    feature_extractor = pipeline('feature-extraction', model=model, tokenizer=tokenizer, truncation=True)

    BERT_train.columns

    #Encodings for tokenisation
    import numpy as np
    X_train = np.array([feature_extractor(text)[0][0] for text in BERT_train['text']])
    X_val = np.array([feature_extractor(text)[0][0] for text in BERT_val['text']])

    #Vectorizer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn import preprocessing

    clf = LogisticRegression(random_state=0).fit(X_train, BERT_train['meta.genre'])

    val_predictions = clf.predict(X_val)
    evaluate(BERT_val, val_predictions)

if __name__ == "__main__":
    main()
