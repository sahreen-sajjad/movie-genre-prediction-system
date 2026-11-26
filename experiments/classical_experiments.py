"""Historical notebook experiment; not independently validated."""
from src.data import load_data, split_data

def main():
    import pandas as pd
    final_df = load_data()
    train, test, val = split_data(final_df)
    conversations = final_df
    top_genres = conversations['meta.genre'].value_counts().head(25)

    top_genres

    single_genres = conversations['meta.genre'].str.count(',') == 0 
    single_genre_movies = conversations.loc[single_genres].copy()
    unique_genres = single_genre_movies['meta.genre'].unique()
    genre_counts = single_genre_movies['meta.genre'].value_counts()
    print(genre_counts)

    print("Frequency in stratified_train:")
    print(train['meta.genre'].value_counts())

    print("\nFrequency in stratified_test:")
    print(test['meta.genre'].value_counts())

    print("\nFrequency in stratified_validation:")
    print(val['meta.genre'].value_counts())

    import matplotlib.pyplot as plt
    train['meta.genre'].value_counts().plot(kind='bar', title="Frequency in training set")
    plt.show()
    test['meta.genre'].value_counts().plot(kind='bar', title="Frequency in test set")
    plt.show()
    val['meta.genre'].value_counts().plot(kind='bar', title="Frequency in validation set")
    plt.show()

    import spacy
    nlp = spacy.load("en_core_web_sm")

    import string
    def tokenize_rulebased(text):
      prev_c = ''
      tokens = []

      for c in text:
        # Ignore characters that are whitespace (e.g. spaces, tabs, etc)
        if not c in string.whitespace and not c in string.punctuation:
          if prev_c in string.whitespace:
            is_new_token = True
          elif prev_c in string.ascii_letters and c not in string.ascii_letters:
            is_new_token = True
          elif prev_c in string.punctuation and c in string.ascii_letters:
            is_new_token = True
          elif prev_c in string.digits and c in string.punctuation:
            is_new_token = True
          elif prev_c in string.punctuation and c in string.digits:
            is_new_token = True
          else:
            is_new_token = False
          if is_new_token:
            tokens.append(c)
          else: 
            tokens[-1] += c
        prev_c = c # Track what the previous character is

      return tokens  



    def stemming(tokens): 
      stemmed_tokens = []
      for token in tokens: 
        if token.endswith('ing'):
          token = token[0:-3]
        if token.endswith('ed'):
          token = token[0:-2]
        if token.endswith('s') and tokens != "is":
          token = token[0:-1]
        if token.endswith('izer'):
          token = token[0:-1]
        if token.endswith('able'):
          token = token[0:-4]
        stemmed_tokens.append(token)
      return stemmed_tokens

    import nltk
    from nltk.corpus import stopwords #remove stopwords using NLTK library

    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

    def remove_stopwords(tokens):
        return [token for token in tokens if token.lower() not in stop_words]

    def make_vocabulary(corpus): #make vocabulary
      dict = {}
      count = 0
      for i in corpus:
        for j in i:
          if j not in dict:
            dict[j] = count
            count = count+1    
      return dict

    def doc_frequency(corpus): #calculate document frequency for all tokens
      unique_tokens = []
      freq = Counter()
      for document in corpus:
        if document not in unique_tokens:
          unique_tokens.append(document)
          for token in document:
            freq[token] = freq[token]+1
      return freq



    import math
    from collections import Counter

    def make_tfidf_sparse(tokens, vocab, doc_freq, N): #Create a sparse vector representation
      sparse = {}
      count = Counter(tokens)
      for token, frequency in count.items():
        if token not in vocab:
          continue
        i = vocab[token]
        if frequency == 0:
          term_freq = 0
        else:
          term_freq = 1 + math.log10(frequency)
        idf = math.log10(N/doc_freq[token])
        sparse[i] = term_freq * idf
      return sparse

    cluster_train = train.copy()

    cluster_train['tokens'] = cluster_train['text'].apply(tokenize_rulebased) 
    cluster_train['tokens'] = cluster_train['tokens'].apply(stemming) 
    cluster_train['tokens'] = cluster_train['tokens'].apply(remove_stopwords)

    train_vocab = make_vocabulary(cluster_train['tokens'])
    train_docfreq = doc_frequency(cluster_train['tokens'])
    N = len(cluster_train['tokens'])
    train_tfidf_vectors = [ make_tfidf_sparse(pt, train_vocab, train_docfreq, N) for pt in cluster_train['tokens'] ]

    import numpy as np

    def sparse_dot_prod(sv1, sv2):
        dot_product = 0
        common_indices = set(sv1.keys()) & set(sv2.keys())
        for index in common_indices:
            dot_product += sv1[index] * sv2[index]
        return dot_product

    def sparse_cosine_similarity(sv1, sv2):
      squares = []
      for index,val in sv1.items():
        squares.append(val * val)
      d1 = math.sqrt(sum(squares))
      squares2 = []
      for index,val in sv2.items():
        squares2.append(val * val)
      d2 = math.sqrt(sum(squares))
      if d1 == 0 or d2 == 0:
        return 0
      else:
        sim = sparse_dot_prod(sv1, sv2) / (d1 * d2)
        return sim



    def get_cluster_ids(data_points, centers):
        cluster_ids = []
        for data_point in data_points:
            similarities = [sparse_cosine_similarity(data_point, center) for center in centers]
            cluster_ids.append(similarities.index(max(similarities)))
        return cluster_ids

    import random
    import numpy as np
    import random
    import numpy as np

    def kmeans(cluster_no, vocabulary, vectors):
        iter = 0

        np.random.seed(42)
        centroids = np.random.randn(cluster_no, len(vocabulary)) #Clusters are initialized randomly (Step 1)
        centroids = [dict(zip(range(len(vocabulary)), centroid)) for centroid in centroids]

        cluster_assignments = []
        #Here, each vector is assigned to its closest centroid (Step 2)
        for vector in vectors:
          similarities = [sparse_cosine_similarity(vector, centroid) for centroid in centroids]
          closest_centroid = similarities.index(max(similarities))
          cluster_assignments.append(closest_centroid)

        prev_cluster_assignments = cluster_assignments

        #Until the model converges (Repeating the steps)
        while iter == 0 or count_diff > 0:
            num_clusters = len(set(cluster_assignments))
            cluster_vectors = []
            for i in range(num_clusters):
              cluster_vectors.append([])
            for vector, cluster_id in zip(vectors, cluster_assignments):
                cluster_vectors[cluster_id].append(vector)
            #Here, new centroids are calculated based on the closest vector (Step 3)
            new_centroids = [dict() for i in range(num_clusters)]
            for i in range(num_clusters):
              for key in set().union(*[v.keys() for v in cluster_vectors[i]]):
                value_sum = sum(v.get(key, 0) for v in cluster_vectors[i])
                new_centroids[i][key] = value_sum / len(cluster_vectors[i])

            cluster_assignments = get_cluster_ids(vectors, new_centroids)
            count_diff = sum([1 for a, b in zip(cluster_assignments, prev_cluster_assignments) if a != b])
            prev_cluster_assignments = cluster_assignments
            iter += 1

        return cluster_assignments, new_centroids

    #Implementing K-means 
    cluster_ids, centroids = kmeans(5, train_vocab, train_tfidf_vectors)


    cluster_docs = {}
    for i in set(cluster_ids):
        cluster_docs[i] = []
    cluster_train['Cluster'] = cluster_ids
    clusters = cluster_train.groupby(['Cluster', 'meta.genre'])
    clusters = clusters.size()
    clusters = clusters.unstack(level='meta.genre')

    clusters

    import random

    random.seed(42)
    document_dict = dict(enumerate(cluster_train['tokens']))

    cluster_docs = {}
    for cluster_id in set(cluster_ids):
      indices = []
      for i, c_id in enumerate(cluster_ids):
        if c_id == cluster_id:
          indices.append(i)
        cluster_docs[cluster_id] = indices


    centroid_tuples = list(enumerate(centroids))

    for cluster_id in cluster_docs:
        docs = cluster_docs[cluster_id]
        print(f'Cluster {cluster_id}:\n{len(docs)} documents assigned to this cluster')
        print()
        print('Sample documents:')
        sample_docs = random.sample(docs, min(5, len(docs)))
        for doc_id in sample_docs:
            document = document_dict[doc_id]
            print(document)

        print()

    top_tokens = []
    for cluster_id, centroid in enumerate(centroids):
      sorted_tokens = sorted(centroid.items(), key=lambda x: -x[1])
      sorted_tokens = sorted_tokens[:5]
      top_tokens.append((cluster_id, sorted_tokens))


    for cluster_id in cluster_docs:
        print(f'Cluster {cluster_id}:')
        docs = cluster_docs[cluster_id]
        for cluster_tuple in top_tokens:
          if cluster_tuple[0] == cluster_id:
            top_tokens_cluster = cluster_tuple[1]
            break


        print(f'Top 5 tokens with highest magnitude for cluster {cluster_id}:')
        for token_id, score in top_tokens_cluster:
            for word, id in train_vocab.items():
              if id == token_id:
                token = word
                break
            print(f'{token}: {score:.4f}')
        print()


    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sns.heatmap(clusters, ax=ax, cmap='YlGnBu', fmt='.0f', annot=True)

    ax.set_xlabel('Genre')
    ax.set_ylabel('Cluster')
    plt.show()

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    def evaluate(val, val_predictions):
        val_precision, val_recall, val_f1, support = precision_recall_fscore_support(val['meta.genre'], val_predictions, average='macro')
        val_accuracy = (val['meta.genre'] == val_predictions).mean()
        print(f'Validation accuracy: {val_accuracy:.3f}')
        print(f'Validation precision (macro-averaged): {val_precision:.3f}')
        print(f'Validation recall (macro-averaged): {val_recall:.3f}')
        print(f'Validation F1-score (macro-averaged): {val_f1:.3f}')
        label_dist = pd.Series(val_predictions).value_counts()
        print(label_dist)

        plt.bar(label_dist.index, label_dist.values)
        plt.title("Label Distribution")
        plt.xlabel("Genre")
        plt.ylabel("Frequency")
        plt.show()


    from sklearn.dummy import DummyClassifier

    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(train['text'], train['meta.genre'])
    prediction = dummy_clf.predict(train[['text']])

    val_predictions = dummy_clf.predict(val[['text']])
    val_accuracy = accuracy_score(val['meta.genre'], val_predictions)
    evaluate(val, val_predictions)

    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_clf.fit(train['text'], train['meta.genre'])
    prediction = dummy_clf.predict(train[['text']])

    val_predictions = dummy_clf.predict(val[['text']])
    val_accuracy = accuracy_score(val['meta.genre'], val_predictions)
    evaluate(val, val_predictions)

    classifiers_train = train.copy()
    classifiers_test = test.copy()
    classifiers_val = val.copy()

    #import spacy
    #import nltk

    #nltk.download('punkt')  # download NLTK's punkt tokenizer

    # download and load spaCy's English language model
    #spacy.cli.download('en_core_web_sm')
    #nlp = spacy.load('en_core_web_sm')

    #def tokenize_text(text):
        #return [token.text for token in nlp(text)]

    import spacy
    import pandas as pd

    nlp = spacy.load('en_core_web_sm')

    def tokenize_text(text):
        return [token.text for token in nlp(text)]

    classifiers_train['tokens'] = classifiers_train['text'].apply(tokenize_text)
    classifiers_test['tokens'] = classifiers_test['text'].apply(tokenize_text)
    classifiers_val['tokens'] = val['text'].apply(tokenize_text)

    #import pandas as pd
    #from nltk.stem import PorterStemmer

    #stemmer = PorterStemmer()

    #def stem_text(tokens):
        #stemmed_tokens = []
        #for token in tokens:
            #stemmed_tokens.append(stemmer.stem(token))
        #return stemmed_tokens

    import pandas as pd
    from nltk.stem import SnowballStemmer

    stemmer = SnowballStemmer('english')

    def stem_text(tokens):
        stemmed_tokens = []
        for token in tokens:
            stemmed_tokens.append(stemmer.stem(token))
        return stemmed_tokens

    classifiers_train['tokens'] = classifiers_train['tokens'].apply(stem_text)
    classifiers_test['tokens'] = classifiers_test['tokens'].apply(stem_text)
    classifiers_val['tokens'] = classifiers_val['tokens'].apply(stem_text)

    import spacy

    nlp = spacy.load('en_core_web_sm')  

    def lemmatize_text(text):
        lemmatized_tokens = []
        for token in text:
            lemmatized_tokens.append(nlp(token)[0].lemma_)
        return lemmatized_tokens

    classifiers_train['tokens'] = classifiers_train['tokens'].apply(lemmatize_text)
    classifiers_test['tokens'] = classifiers_test['tokens'].apply(lemmatize_text)
    classifiers_val['tokens'] = classifiers_val['tokens'].apply(lemmatize_text)

    import pandas as pd
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


    def remove_stopwords(tokens):
      tok = []
      for token in tokens:
        if token not in stop_words:
          tok.append(token)
      return tok


    classifiers_train['tokens'] = classifiers_train['tokens'].apply(remove_stopwords)
    classifiers_test['tokens'] = classifiers_test['tokens'].apply(remove_stopwords)
    classifiers_val['tokens'] = classifiers_val['tokens'].apply(remove_stopwords)

    # Convert token lists to space-separated strings
    classifiers_train['text'] = classifiers_train['tokens'].apply(lambda tokens: ' '.join(tokens))
    classifiers_val['text'] = classifiers_val['tokens'].apply(lambda tokens: ' '.join(tokens))

    #logistic regression one hot encoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression


    enc = OneHotEncoder(handle_unknown='ignore')
    X_train = enc.fit_transform(classifiers_train[['text']])
    y_train = classifiers_train['meta.genre']
    X_val = enc.transform(classifiers_val[['text']])
    y_val = classifiers_val['meta.genre']


    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    val_predictions = clf.predict(X_val)
    evaluate(classifiers_val, val_predictions)

    #Logistic Regression with Count Vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(classifiers_train['text'])
    y_train = classifiers_train['meta.genre']
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    X_val = vectorizer.transform(classifiers_val['text'])
    y_val = classifiers_val['meta.genre']
    val_predictions = clf.predict(X_val)

    evaluate(classifiers_val, val_predictions)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(classifiers_train['text'])
    y_train = classifiers_train['meta.genre']
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    X_val = vectorizer.transform(classifiers_val['text'])
    y_val = classifiers_val['meta.genre']
    val_predictions = clf.predict(X_val)

    evaluate(classifiers_val, val_predictions)


    from sklearn.preprocessing import OneHotEncoder
    from sklearn.svm import SVC


    enc = OneHotEncoder(handle_unknown='ignore')
    X_train = enc.fit_transform(classifiers_train[['text']])
    y_train = classifiers_train['meta.genre']
    X_val = enc.transform(classifiers_val[['text']])
    y_val = classifiers_val['meta.genre']

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    val_predictions = clf.predict(X_val)
    evaluate(classifiers_val, val_predictions)


    import matplotlib.pyplot as plt

    _, _, f1_scores, classes = precision_recall_fscore_support(y_val, val_predictions, labels=clf.classes_)

    fig, ax = plt.subplots()
    ax.bar(classes, f1_scores, color='green', width = 10)
    ax.set_xlabel('Class')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score against Classes')
    plt.show()


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(classifiers_train['text'])
    y_train = classifiers_train['meta.genre']
    clf = SVC(kernel='rbf',random_state=42, probability=True)
    clf.fit(X_train, y_train)

    X_val = vectorizer.transform(classifiers_val['text'])
    y_val = classifiers_val['meta.genre']
    val_predictions = clf.predict(X_val)
    evaluate(classifiers_val, val_predictions)

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier

    enc = OneHotEncoder(handle_unknown='ignore')
    X_train = enc.fit_transform(classifiers_train[['text']])
    y_train = classifiers_train['meta.genre']
    X_val = enc.transform(classifiers_val[['text']])
    y_val = classifiers_val['meta.genre']


    clf = RandomForestClassifier().fit(X_train, y_train)
    val_predictions = clf.predict(X_val)
    evaluate(classifiers_val, val_predictions)

    from sklearn.metrics import precision_recall_fscore_support

    def evaluate_hyperparameters(val, val_predictions, search):
        val_precision, val_recall, val_f1, support = precision_recall_fscore_support(val['meta.genre'], val_predictions, average='macro')
        val_accuracy = (val['meta.genre'] == val_predictions).mean()
        print(f'Validation accuracy after tuning feature: {val_accuracy:.3f}')
        print(f'Validation precision (macro-averaged) after tuning feature: {val_precision:.3f}')
        print(f'Validation recall (macro-averaged) after tuning feature: {val_recall:.3f}')
        print(f'Validation F1-score (macro-averaged) after tuning feature: {val_f1:.3f}')
        print(f'Best parameters after tuning feature: {search.best_params_}')

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    param_grid_1 = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    clf = LogisticRegression(max_iter=10000)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search = GridSearchCV(pipe, param_grid_1, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])
    best_model = search.best_estimator_
    val_predictions = best_model.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions, search)

    #only tuning sublinear_tf

    param_grid_2 = {
        'tfidf__sublinear_tf': [True, False]
    }

    clf = LogisticRegression(max_iter=10000)
    pipe_2 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search = GridSearchCV(pipe_2, param_grid_2, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])
    best_model = search.best_estimator_
    val_predictions = best_model.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions, search)

    #only tuning max_features

    param_grid_2 = {
        'tfidf__max_features': [None, 50000]
    }

    clf = LogisticRegression(max_iter=10000)
    pipe_2 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search = GridSearchCV(pipe_2, param_grid_2, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])
    best_model = search.best_estimator_
    val_predictions = best_model.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions, search)

    param_grid_3 = {
        'clf__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
    }

    clf = LogisticRegression(max_iter=10000)
    pipe_3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search_3 = GridSearchCV(pipe_3, param_grid_3, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])

    best_model_3 = search_3.best_estimator_
    val_predictions = best_model_3.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions, search_3)

    param_grid_3 = {
        'clf__multi_class': ['ovr', 'multinomial']
    }

    clf = LogisticRegression(max_iter=10000)
    pipe_3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search_3 = GridSearchCV(pipe_3, param_grid_3, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])
    best_model_3 = search_3.best_estimator_
    val_predictions_3 = best_model_3.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions_3, search_3)


    param_grid_3 = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)]
    }


    clf = LogisticRegression(max_iter=10000)
    pipe_3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search_3 = GridSearchCV(pipe_3, param_grid_3, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])

    best_model_3 = search_3.best_estimator_
    val_predictions_3 = best_model_3.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions_3, search_3)



    param_grid_3 = {
        'tfidf__max_df': [0.5] #ignore terms that appear in more than 50% of the documents
    }

    clf = LogisticRegression(max_iter=10000)
    pipe_3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search_3 = GridSearchCV(pipe_3, param_grid_3, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])

    best_model_3 = search_3.best_estimator_
    val_predictions_3 = best_model_3.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions_3, search_3)

    param_grid_3 = {
        'tfidf__min_df': [1, 5, 10]
    }


    clf = LogisticRegression(max_iter=10000)
    pipe_3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', clf)
    ])

    search_3 = GridSearchCV(pipe_3, param_grid_3, cv=5).fit(classifiers_train['text'], classifiers_train['meta.genre'])

    best_model_3 = search_3.best_estimator_
    val_predictions_3 = best_model_3.predict(classifiers_val['text'])
    evaluate_hyperparameters(classifiers_val, val_predictions_3, search_3)

    vectorizer = TfidfVectorizer(ngram_range = (1, 2))
    clf = LogisticRegression(C=1000)
    pipe = Pipeline([
        ('tfidf', vectorizer),
        ('clf', clf)
    ])

    pipe.fit(classifiers_train['text'], classifiers_train['meta.genre'])
    val_predictions = pipe.predict(classifiers_val['text'])
    evaluate(classifiers_val, val_predictions)

if __name__ == "__main__":
    main()
