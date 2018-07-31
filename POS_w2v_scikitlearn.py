#!/usr/bin/env python

'''
Modified from https://nlpforhackers.io/training-pos-tagger/
'''

import nltk, gensim
import numpy as np

tagged_sentences = nltk.corpus.treebank.tagged_sents()

# print(tagged_sentences[0])
# print("Tagged sentences: ", len(tagged_sentences))
# print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]
# print(untag(tagged_sentences[0]))

def untagged_whole(tagged_sentences):
    sent_list = []
    for s in tagged_sentences:
        sent_list.append(untag(s))
    return sent_list

def train_w2v(sent_list, size=100):
    model = gensim.models.Word2Vec(sent_list, min_count=1,size=size)
    model.delete_temporary_training_data(replace_word_vectors_with_normalized=True)
    return model

def features(sentence, index, size, w2vmodel):
    """ sentence: [w1, w2, ...], index: the index of the word """
    input_vec = np.empty((0, 0), float)
    if index == 0:
        input_vec = np.append(input_vec, np.array([0] * size))
    else:
        input_vec = np.append(input_vec, w2vmodel.wv[sentence[index - 1]])

    input_vec = np.append(input_vec,w2vmodel.wv[sentence[index]])

    if index == len(sentence) - 1:
        input_vec = np.append(input_vec, np.array([1] * size))
    else:
        input_vec = np.append(input_vec, w2vmodel.wv[sentence[index + 1]])
    return input_vec.reshape(3*size,1)


# Split the dataset for training and testing
cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

# print(len(training_sentences))  # 2935
# print(len(test_sentences))  # 979

vector_size = 100
#train the word2vec model
model = train_w2v(untagged_whole(tagged_sentences), vector_size)


def transform_to_dataset(tagged_sentences, model):
    X, y = [], []
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index, vector_size, model))
            y.append(tagged[index][1])
    X = np.array(X).reshape(len(X),3 * vector_size)
    return X, y

X, y = transform_to_dataset(training_sentences, model)

# print(X[0:3])
# print("lenX:",len(X))

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def NNmodel(h1=5000,h2=3000):
    clf = Pipeline([('classifier',
                     MLPClassifier(hidden_layer_sizes=(h1,h2,), activation='relu', max_iter=2000, alpha=1e-4,
                                   solver='adam', learning_rate='adaptive', verbose=True, tol=0.0001, random_state=5,
                                   learning_rate_init=.001))])
    return clf

result=[]
for i in [3000,4000,5000,6000,7000]:
    clf = NNmodel(5000,i)
    clf.fit(X[:50000], y[:50000])
    print('Training completed')
    X_test, y_test = transform_to_dataset(test_sentences, model)
    acc = clf.score(X_test, y_test)
    result.append((5000,i,acc))
    print("hidden1: ", 5000)
    print("hidden2: ", i)
    print("Accuracy:", acc)

print(result)

