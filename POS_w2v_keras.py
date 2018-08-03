#!/usr/bin/env python

'''
POS_w2v_keras.py
Min Chen <mc43@iu.edu>
Project: Deep Learning and POS tagging

Corpus: Treebank from NLTK
Libary: keras
Model: Neural Network
Word Embedding: Yes

Last Updated by Min Chen - Aug 2,2018

'''


import nltk, gensim, os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils, plot_model
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

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

CUSTOM_SEED = 42


def transform_to_dataset(tagged_sentences, model):
    X, y = [], []
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index, vector_size, model))
            y.append(tagged[index][1])
    X = np.array(X).reshape(len(X),3 * vector_size)
    return X, y


def build_model(input_dim, hidden_neurons1, hidden_neurons2,output_dim):
    """
    Construct, compile and return a Keras model which will be used to fit/predict
    """
    model = Sequential([
        Dense(hidden_neurons1, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons2),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc, save_figure_path):
    """ Plot model loss and accuracy through epochs. """

    green = '#72C29B'
    orange = '#FFA577'

    with plt.xkcd():
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
        ax1.plot(range(1, len(train_loss) + 1), train_loss, green, linewidth=5,
                 label='training')
        ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, orange,
                 linewidth=5, label='validation')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('loss')
        ax1.tick_params('y')
        ax1.legend(loc='upper right', shadow=False)
        ax1.set_title('Model loss through #epochs', fontweight='bold')

        ax2.plot(range(1, len(train_acc) + 1), train_acc, green, linewidth=5,
                 label='training')
        ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, orange,
                 linewidth=5, label='validation')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('accuracy')
        ax2.tick_params('y')
        ax2.legend(loc='lower right', shadow=False)
        ax2.set_title('Model accuracy through #epochs', fontweight='bold')

    plt.tight_layout()
    plt.show()
    fig.savefig(save_figure_path)
    plt.close(fig)


if __name__ == '__main__':
    # Ensure reproducibility
    vector_size = 100
    np.random.seed(CUSTOM_SEED)

    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    print(tagged_sentences[0])
    print("Tagged sentences: ", len(tagged_sentences))
    print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))

    model = train_w2v(untagged_whole(tagged_sentences), vector_size)

    # We use approximately 60% of the tagged sentences for training,
    # 20% as the validation set and 20% to evaluate our model.
    train_test_cutoff = int(.80 * len(tagged_sentences))
    training_sentences = tagged_sentences[:train_test_cutoff]
    testing_sentences = tagged_sentences[train_test_cutoff:]

    train_val_cutoff = int(.25 * len(training_sentences))
    validation_sentences = training_sentences[:train_val_cutoff]
    training_sentences = training_sentences[train_val_cutoff:]

    # For training, validation and testing sentences, we split the
    # attributes into X (input variables) and y (output variables).
    X_train, y_train = transform_to_dataset(training_sentences,model)
    X_test, y_test = transform_to_dataset(testing_sentences,model)
    X_val, y_val = transform_to_dataset(validation_sentences,model)

    # Fit LabelEncoder with our list of classes
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_test + y_val)

    # Encode class values as integers
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_val = label_encoder.transform(y_val)

    # Convert integers to dummy variables (one hot encoded)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)

    # Set model parameters
    model_params = {
        'build_fn': build_model,
        'input_dim': X_train.shape[1],
        'hidden_neurons1': 4000,
        'hidden_neurons2': 4000,
        'output_dim': y_train.shape[1],
        'epochs': 20,
        'batch_size': 256,
        'verbose': 1,
        'validation_data': (X_val, y_val),
        'shuffle': True
    }

    # Create a new sklearn classifier
    clf = KerasClassifier(**model_params)

    # Finally, fit our classifier
    hist = clf.fit(X_train, y_train)

    # Plot model performance
    plot_model_performance(
        train_loss=hist.history.get('loss', []),
        train_acc=hist.history.get('acc', []),
        train_val_loss=hist.history.get('val_loss', []),
        train_val_acc=hist.history.get('val_acc', []),
        save_figure_path='results/POS_w2v_keras/model_performance.png'
    )

    # Evaluate model accuracy
    score = clf.score(X_test, y_test, verbose=0)
    print('model accuracy: {}'.format(score))

    # Visualize model architecture
    plot_model(clf.model, to_file='results/POS_w2v_keras/model_structure.png', show_shapes=True)

    # Finally save model
    clf.model.save('results/POS_w2v_keras/keras_mlp.h5')
