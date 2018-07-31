'''
Modified from https://techblog.cdiscount.com/part-speech-tagging-tutorial-keras-deep-learning-library/

'''


import nltk, os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {'word': sentence[index], 'is_first': index == 0, 'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]}



def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

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
    CUSTOM_SEED = 42
    np.random.seed(CUSTOM_SEED)
    CORPUS = 'Treebank'
    # CORPUS = 'Brown'
    PATH = 'results/POS_keras_brown/'

    if CORPUS == 'Treebank':
        tagged_sentences = nltk.corpus.treebank.tagged_sents()
    else:
        tagged_sentences = nltk.corpus.brown.tagged_sents()
    print(tagged_sentences[0])
    print("Tagged sentences: ", len(tagged_sentences))


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
    X_train, y_train = transform_to_dataset(training_sentences)
    X_test, y_test = transform_to_dataset(testing_sentences)
    X_val, y_val = transform_to_dataset(validation_sentences)

    # Fit our DictVectorizer with our set of features
    dict_vectorizer = DictVectorizer(sparse=False)
    dict_vectorizer.fit(X_train + X_test + X_val)

    # Convert dict features to vectors
    X_train = dict_vectorizer.transform(X_train)
    X_test = dict_vectorizer.transform(X_test)
    X_val = dict_vectorizer.transform(X_val)

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

    model = Sequential()
    model.add(Dense(400, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("model fitting sequential")
    model.summary()

    x = model.fit(X_train, y_train,
                  batch_size=256,
                  epochs=8,
                  validation_data=(X_val, y_val),
                  shuffle=True,
                  verbose=1)


    if not os.path.exists(PATH):
        print('MAKING DIRECTORY to save model file')
        os.makedirs(PATH)

    # Plot model performance
    plot_model_performance(
        train_loss=x.history.get('loss', []),
        train_acc=x.history.get('acc', []),
        train_val_loss=x.history.get('val_loss', []),
        train_val_acc=x.history.get('val_acc', []),
        save_figure_path=PATH + 'model_performance.png'
    )

    # Visualize model architecture
    plot_model(model, to_file=PATH + 'model_structure.png', show_shapes=True)

    test_results = model.evaluate(X_test, y_test, verbose=1)
    print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

    print("see results in " + PATH)
