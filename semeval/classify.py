import sys
import math
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import *
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing import *
import keras.preprocessing.text as kpt
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras import optimizers
import fasttext
from evaluation_metrics import compute_confusion_matrix


parser = ArgumentParser();
# NOTE: input format is the same as the written output of json_to_txt.py
parser.add_argument("-tr", "--training_filename");
parser.add_argument("-ts", "--test_filename");  # can be the same with the training set
parser.add_argument("-mo", "--model_name", help="RandomForest runs with TFIDF, FastText with words/ngrams and no pretrained embeddings, "
                            "Keras networks can run with one-hot-encoding and potentially pret-rained embeddings");
parser.add_argument("-em", "--embedding_type", help="file path to Glove, or file path to FastText");
parser.add_argument("-sc", "--scale", help="currently used only for the Keras networks");
parser.add_argument("-mt", "--maximum_frequent_terms", help="for_tfidf");
parser.add_argument("-mw", "--maximum_word_number", help="aka_input_dimension");
parser.add_argument("-ed", "--embedding_dimension", help="aka_output_dimension");
parser.add_argument("-ms", "--maximum_sequence_length");
parser.add_argument("-bs", "--batch_size"); # batch_size could be math.ceil(len(X_train)*0.1);
parser.add_argument("-ep", "--epochs");
parser.add_argument("-op", "--optimizer")
# parameter example:
# -tr "data.txt" -ts "data.txt" -em "/glove.42B.300d.txt" -mt "1000" -mw "1000" -ms "1000" -bs "32" -ep "10" -op "sgd"
# TODO: params for lowercase, stopwords, stemming, long docs, n-grams
# TODO: params for learning rate, momentum, decay
args = parser.parse_args();

x_train = list(); # training_docs
y_train = list(); # training_targets
x_test = list(); # test_docs
y_test = list(); # test_targets


def load_data():
    global x_train
    global x_test
    global y_train
    global y_test
    delimiter = "__label__";
    with open(args.training_filename, "r") as inputf:
        for line in inputf.readlines():
            text = line.split(delimiter)[0];
            x_train.append(text);
            label = line.split(delimiter)[1].replace("\n", "");
            y_train.append(label);
    with open(args.test_filename, "r") as inputf:
        for line in inputf.readlines():
            text = line.split(delimiter)[0];
            x_test.append(text);
            label = line.split(delimiter)[1].replace("\n", "");
            y_test.append(label);
    print("x_train size: {}".format(len(x_train)));
    print("x_test size: {}".format(len(x_test)));


def process_labels():
    global y_train
    global y_test
    y_train = [0 if x == "false" else 1 for x in y_train];
    y_train = to_categorical(y_train, num_classes=2);  # [1. 0.] ==> [[0. 1.], [1. 0.]]
    y_test = [0 if x == "false" else 1 for x in y_test];
    y_test_orginal = y_test;
    y_test = to_categorical(y_test, num_classes=2);
    print("y_train shape: {}".format(y_train.shape));
    print("y_test shape: {}".format(y_test.shape));
    return y_test_orginal;


def compute_tf_idf():
    count_vector = CountVectorizer(max_features=int(float(args.maximum_frequent_terms)), stop_words='english');
    x_train_vec = count_vector.fit_transform(x_train);
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_vec);
    x_train_tf = tf_transformer.transform(x_train_vec);
    x_test_vec = count_vector.fit_transform(x_test);
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_test_vec);
    x_test_tf = tf_transformer.transform(x_test_vec);
    print("x_train_tf shape: {}\n".format(x_train_tf.shape));
    print("x_test_tf shape: {}\n".format(x_test_tf.shape));
    return x_train_tf, x_test_tf;


def compute_one_hot_encoding_old():
    def convert_text_to_index_array(text, dictionary, remove_stopwords):
        stopwords_set = set(stopwords.words("english"));
        if remove_stopwords:
            words = [word for word in kpt.text_to_word_sequence(text) if word not in stopwords_set];
        else:
            words = kpt.text_to_word_sequence(text);
        return [dictionary[word] for word in words];
    tokenizer = kpt.Tokenizer(num_words=int(float(args.maximum_word_number)));
    tokenizer.fit_on_texts(x_train);
    word_idx_train = tokenizer.word_index;
    print("Training set dictionary size: {}".format(len(word_idx_train)));
    # every text corresponds to a sequence of word indices
    x_train_widx = [];
    for data_instance in x_train:
        word_indices = convert_text_to_index_array(data_instance, word_idx_train, remove_stopwords=True);
        x_train_widx.append(word_indices);
    x_train_widx = np.asarray(x_train_widx);
    x_train_oh = tokenizer.sequences_to_matrix(x_train_widx, mode='binary');
    tokenizer = kpt.Tokenizer(num_words=int(float(args.maximum_word_number)));
    tokenizer.fit_on_texts(x_test);
    word_idx_test = tokenizer.word_index;
    print("Test set dictionary size: {}".format(len(word_idx_test)));
    x_test_widx = [];
    for data_instance in x_test:
        word_indices = convert_text_to_index_array(data_instance, word_idx_test, remove_stopwords=True);
        x_test_widx.append(word_indices);
    x_test_widx = np.asarray(x_test_widx);
    x_test_oh = tokenizer.sequences_to_matrix(x_test_widx, mode='binary');
    if args.scale == "1":
        scaler = preprocessing.StandardScaler().fit(x_train_oh);
        x_train_oh = scaler.transform(x_train_oh);
        x_test_oh = scaler.transform(x_test_oh);
    # change numpy arrays to list of lists
    x_train_oh = x_train_oh.tolist();
    x_test_oh = x_test_oh.tolist();
    # ensure that all sequences in the list have the same length
    x_train_oh = sequence.pad_sequences(x_train_oh, maxlen=int(float(args.maximum_sequence_length)));
    x_test_oh = sequence.pad_sequences(x_test_oh, maxlen=int(float(args.maximum_sequence_length)));
    print("X_train_oh shape: {}".format(x_train_oh.shape));
    print("X_test_oh shape: {}".format(x_test_oh.shape));
    return x_train_oh, x_test_oh, word_idx_train, word_idx_test;


def compute_one_hot_encoding_new():
    t = kpt.Tokenizer(num_words=int(float(args.maximum_word_number)));
    t.fit_on_texts(x_train);
    print("words in training set: {}".format(len(t.word_counts)));
    print("docs in training set: {}".format(t.document_count));
    training_idx = t.word_index;
    print("word frequencies in training set: {}".format(t.word_docs));
    x_train_oh = t.texts_to_matrix(x_train, mode='count');
    t.fit_on_texts(x_test);
    print("words in test set: {}".format(len(t.word_counts)));
    print("docs in test set: {}".format(t.document_count));
    test_idx = t.word_index;
    print("training word index size: {}, test word index size: {}".format(len(training_idx), len(test_idx)));
    x_test_oh = t.texts_to_matrix(x_test, mode='count');
    # if args.scale == "1":
    #     scaler = preprocessing.StandardScaler().fit(x_train_oh);
    #     x_train_oh = scaler.transform(x_train_oh);
    #     x_test_oh = scaler.transform(x_test_oh);
    print("X_train_oh shape: {}".format(x_train_oh.shape));
    print("X_test_oh shape: {}".format(x_test_oh.shape));
    return x_train_oh, x_test_oh, training_idx;


def set_up_embedding_layer(word_idx_training):
    if args.embedding_type == "not-pretrained":
        return Embedding(input_dim=int(float(args.maximum_word_number)),
                         input_length=int(float(args.maximum_sequence_length)),
                         output_dim=int(float(args.embedding_dimension)));
    elif "glove" in args.embedding_type:
        embeddings_matrix = load_glove_embedding(word_idx_training);
        return Embedding(input_dim=(len(word_idx_training) + 1),
                         output_dim=300,
                         weights=[embeddings_matrix],
                         input_length=int(float(args.maximum_sequence_length)),
                         trainable=False);


def load_glove_embedding(word_idx_training):
    embedding_index = dict();
    # only 17849/23928 found
    # TODO: stemming
    with open(args.embedding_type) as f:
        for line in f:
            values = line.split();
            word = values[0];
            if word in word_idx_training.keys():
                coefs = np.asarray(values[1:], dtype='float32');
                embedding_index[word] = coefs;
    print("Number of word vectors: {}".format(len(embedding_index)));
    embedding_matrix = np.zeros((len(word_idx_training) + 1, 300));
    for word, i in word_idx_training.items():
        embedding_vector = embedding_index.get(word);
        if embedding_index.get(word) is not None:
            embedding_matrix[i] = embedding_vector;
    print("Embedding matrix shape: {}".format(embedding_matrix.shape));
    return embedding_matrix;


def build_fully_connected_network(word_idx_training):
    model = Sequential();
    # choose embedding;
    embedding_layer = set_up_embedding_layer(word_idx_training=word_idx_training);
    model.add(embedding_layer);
    # model.add(Dense(32, activation='relu', input_dim=int(float(args.maximum_word_number))));
    model.add(Dense(256, activation='relu'));
    model.add(Dense(128, activation='relu'));
    model.add(Dense(64, activation='relu'));
    # hidden layers
    model.add(Flatten());
    model.add(Dense(2, activation='softmax'));  # alternatively, one output dimension with binary_crossentropy
    return model;


def build_lstm_network(word_idx_training):
    model = Sequential();
    # choose embedding;
    embedding_layer = set_up_embedding_layer(word_idx_training=word_idx_training);
    model.add(embedding_layer);
    model.add(Conv1D(128, 5, activation='relu'));
    model.add(MaxPooling1D(pool_size=4));
    model.add(LSTM(64));
    # hidden layers
    model.add(Dense(2, activation='softmax'));  # alternatively, one output dimension with binary_crossentropy
    return model;


def build_bilstm_network(word_idx_training):
    model = Sequential();
    # choose embedding;
    embedding_layer = set_up_embedding_layer(word_idx_training=word_idx_training);
    model.add(embedding_layer);
    model.add(Bidirectional(LSTM(128)));
    # hidden layers
    model.add(Dense(2, activation='softmax'));  # alternatively, one output dimension with binary_crossentropy
    return model;


def run_keras_model(model):
    # customize optimizer
    if args.optimizer == "sgd":
        sgd = optimizers.SGD(lr=0.001, clipvalue=0.5, momentum=0.0, decay=0.95, nesterov=False);
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']);
    elif args.optimizer == "adam":
        adam = optimizers.Adam(lr=0.0001, decay=0.95);
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']);
    print(model.summary());
    # compile, fit and predict
    model.fit(x=x_train, y=y_train, batch_size=int(float(args.batch_size)), epochs=int(float(args.epochs)),
              shuffle=True);  # validation_split=0.2
    y_pred = model.predict(x_test);
    # print("Check predicted classes per data instance:");
    # for pred in y_pred:
    #     print(pred);
    y_pred_list = list();
    # transform predicted classes to list that will be fed to the evaluation metric func
    for arr in y_pred:
        predicted_class_idx = np.argmax(arr);  # 0 for negative class, 1 for positive class
        if predicted_class_idx == 0:  # [1. 0.] ==> [[0. 1.], [1. 0.]]
            y_pred_list.append(0);
        else:
            y_pred_list.append(1);
    print("Y_train: {}".format(y_train));
    print("Y_test: {}".format(y_test));
    print("Y_pred_list: {}".format(y_pred_list));
    return y_pred_list, y_test, 1, 0


if __name__ == '__main__':

    load_data();
    # Random Forest
    x_train_tf, x_test_tf = compute_tf_idf();
    rf_clf = RandomForestClassifier(n_jobs=2, verbose=1, class_weight="balanced", random_state=0);
    rf_clf.fit(x_train_tf, y_train);
    clf_result = rf_clf.predict(x_test_tf);
    compute_confusion_matrix(pd.DataFrame(data={"predicted": list(clf_result), "actual": y_test}), pos_label='true',
                             neg_label='false');

    # Fasttext
    fasttext_clf = fasttext.supervised(args.training_filename, 'model');
    clf_result = fasttext_clf.predict(x_test, k=len(x_test));
    y_pred = list();
    for result in clf_result:
        y_pred.append(result[0]);
    compute_confusion_matrix(pd.DataFrame(data={"predicted": y_pred, "actual": y_test}), pos_label='true',
                             neg_label='false');

    # Keras models
    x_train_oh, x_test_oh, word_idx_training = compute_one_hot_encoding_new();
    x_train = x_train_oh;
    x_test = x_test_oh;
    model_1 = build_fully_connected_network(word_idx_training);
    model_2 = build_lstm_network(word_idx_training);
    model_3 = build_bilstm_network(word_idx_training);
    y_test_orginal = process_labels();
    Y_pred_list, y_test, pos_label, neg_label = run_keras_model(model_1);
    compute_confusion_matrix(result=pd.DataFrame(data={"predicted": Y_pred_list, "actual": y_test_orginal}),
                             pos_label=pos_label, neg_label=neg_label);
    Y_pred_list, y_test, pos_label, neg_label = run_keras_model(model_2);
    compute_confusion_matrix(result=pd.DataFrame(data={"predicted": Y_pred_list, "actual": y_test_orginal}),
                             pos_label=pos_label, neg_label=neg_label);
    Y_pred_list, y_test, pos_label, neg_label = run_keras_model(model_3);
    compute_confusion_matrix(result=pd.DataFrame(data={"predicted": Y_pred_list, "actual": y_test_orginal}),
                             pos_label=pos_label, neg_label=neg_label);