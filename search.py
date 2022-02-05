import random
import pandas as pd
import joblib
import pickle
from dop_func import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Document:
    def __init__(self, title, text):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']


index = []
vectorizer_1 = joblib.load('vectroizer_1.pkl')
vectorizer_2 = joblib.load('vectroizer_2.pkl')
with open('id_title.pickle', 'rb') as fp:
    id_title = pickle.load(fp)
with open('id_content.pickle', 'rb') as fp:
    id_content = pickle.load(fp)
df_pr1 = pd.read_csv('dataframe_super.csv')
df_pr2 = pd.read_csv('dataframe_2.csv')
X = vectorizer_1.transform(df_pr2.title)
zz = pd.DataFrame(X.toarray(), columns=vectorizer_1.get_feature_names_out())
X = vectorizer_2.transform(df_pr2.history_text)
zz2 = pd.DataFrame(X.toarray(), columns=vectorizer_2.get_feature_names_out())


def build_index():  # предобработка всего
    for i in range(len(df_pr1)):
        try:
            ind = df_pr1.iloc[i].title.index(',')
        except:
            ind = 100
        index.append(Document(df_pr1.iloc[i].title[:ind],
                              df_pr1.iloc[i].history_text[:100]))


def score(query, document):  # принимает строку и документ
    k_title, k_content = 1.5, 0.5
    query = preprocessing(query)
    real_id = {i: document[i] for i in range(len(document))}
    all_id = [i for i in document]
    title_tfidf = zz.iloc[all_id]
    content_tfidf = zz2.iloc[all_id]
    # score у title
    X = vectorizer_1.transform([query])
    qq1 = np.array(pd.DataFrame(X.toarray(), columns=vectorizer_1.get_feature_names_out()))[0]
    X = vectorizer_2.transform([query])
    qq2 = np.array(pd.DataFrame(X.toarray(), columns=vectorizer_2.get_feature_names_out()))[0]
    print(X.toarray().shape)
    score_title = []
    score_content = []
    for i in range(len(title_tfidf)):
        q_title = title_tfidf.iloc[i]
        user_similarities = cosine_similarity([qq1, np.array(q_title)])
        score_title.append((real_id[i], user_similarities[0, 1]))
        # score у content
        q_content = content_tfidf.iloc[i]
        user_similarities = cosine_similarity([qq2, np.array(q_content)])
        score_content.append((real_id[i], user_similarities[0, 1]))
    # print([i[0] for i in sorted(score_title, key=lambda x: x[1])])
    res = []
    proverka = []
    for i in range(len(score_title)):
        koef = score_title[i][1] * k_title + score_content[i][1] * k_content
        res.append((index[score_title[i][0]], koef))
        proverka.append((score_title[i][0],
                         koef))
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    print(sorted(proverka, key=lambda x: x[1], reverse=True)[:30])
    return res


def retrieve(query):  # принимает строку из поиска
    query = preprocessing(query).split()  # нормализация текста
    candidates_1 = set()  # множество наших индексов
    candidates_2 = set()  # множество наших индексов
    for doc in query:
        id1, id2 = 0, 0
        try:
            candidates_1 = candidates_1.union(id_title[doc])
        except:
            pass
        try:
            candidates_2 = candidates_2.union(id_content[doc])
        except:
            pass
    if len(candidates_2) + len(candidates_1) == 0:
        return [0]
    res = []
    for i in sorted(list(set(list(candidates_1) + list(candidates_2)))):
        res.append(i)
    return res
