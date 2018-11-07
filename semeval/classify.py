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
# TODO: params for lowercase, stopwords, stemming, long docs, n-grams
# TODO: params for learning rate, momentum, decay
args = parser.parse_args();

X_train = list(); # training_docs
Y_train = list(); # training_targets
X_test = list(); # test_docs
Y_test = list(); # test_targets

def convert_text_to_index_array(text, dictionary, remove_stopwords):
    stopwords_set = set(stopwords.words("english"));
    if remove_stopwords:
        words = [word for word in kpt.text_to_word_sequence(text) if word not in stopwords_set];
    else:
        words = kpt.text_to_word_sequence(text);
    return [dictionary[word] for word in words];

def compute_one_hot_encoding():
    tokenizer = kpt.Tokenizer(num_words=int(float(args.maximum_word_number)));
    tokenizer.fit_on_texts(X_train);
    word_index_training = tokenizer.word_index;
    print("Collection dictionary size: {}".format(len(word_index_training)));
    # every text corresponds to a sequence of word indices
    X_train_widx = [];
    for data_instance in X_train:
        word_indices = convert_text_to_index_array(data_instance, word_index_training, remove_stopwords=True);
        X_train_widx.append(word_indices);
    print("Created one index array per data instance. {} indices in total.".format(len(X_train_widx)));
    X_train_widx = np.asarray(X_train_widx);
    X_train_oh = tokenizer.sequences_to_matrix(X_train_widx, mode='binary');
    print("Converted indices to one-hot vectors. {} vectors in total.".format(len(X_train_oh)));
    tokenizer = kpt.Tokenizer(num_words=int(float(args.maximum_word_number)));
    tokenizer.fit_on_texts(X_test);
    word_index_testing = tokenizer.word_index;
    print("Collection dictionary size: {}".format(len(word_index_testing)));
    print(word_index_testing.keys());
    X_test_widx = [];
    for data_instance in X_test:
        word_indices = convert_text_to_index_array(data_instance, word_index_testing, remove_stopwords=True);
        X_test_widx.append(word_indices);
    print("Created one index array per data instance. {} indices in total.".format(len(X_test_widx)));
    X_test_widx = np.asarray(X_test_widx);
    X_test_oh = tokenizer.sequences_to_matrix(X_test_widx, mode='binary');
    print("Converted indices to one-hot vectors. {} vectors in total.".format(len(X_test_oh)));
    print("Data are split: train {}, test {}".format(X_train_oh.shape, X_test_oh.shape));
    if args.scale == "1":
        scaler = preprocessing.StandardScaler().fit(X_train_oh);
        X_train_oh = scaler.transform(X_train_oh);
        X_test_oh = scaler.transform(X_test_oh);
        # print("X train widx mean: {}".format(scaler.mean_));
        # print("X train widx scale: {}".format(scaler.scale_));
    # change numpy arrays to list of lists
    X_train_oh = X_train_oh.tolist();
    X_test_oh = X_test_oh.tolist();
    # ensure that all sequences in the list have the same length
    X_train_oh = sequence.pad_sequences(X_train_oh, maxlen=int(float(args.maximum_sequence_length)));
    X_test_oh = sequence.pad_sequences(X_test_oh, maxlen=int(float(args.maximum_sequence_length)));
    return X_train_oh, X_test_oh;

def compute_tf_idf():
    count_vector = CountVectorizer(max_features=int(float(args.maximum_frequent_terms)), stop_words='english');  # max_df=0.95, min_df=2
    X_train_vec = count_vector.fit_transform(X_train);
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_vec);
    X_train_tf = tf_transformer.transform(X_train_vec);
    X_test_vec = count_vector.fit_transform(X_test);
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_test_vec);
    X_test_tf = tf_transformer.transform(X_test_vec);
    print("TFIDF: Training feature shape: {}\n".format(X_train_tf.shape));
    print("TFIDF: Test feature shape: {}\n".format(X_test_tf.shape));
    return X_train_tf, X_test_tf;

def load_glove_embeddings():
    embeddings_index = {}
    with open(args.embedding_type) as f:
        for line in f:
            values = line.split();
            word = values[0];
            if word in word_index_training.keys() or word in word_index_testing.keys():
                coefs = np.asarray(values[1:], dtype='float32');
                embeddings_index[word] = coefs;
    print('Found %s word vectors.' % len(embeddings_index));
    embedding_matrix = np.zeros((len(word_index_training) + 1, int(float(args.embedding_dimension))));
    for word, i in word_index_training.items():
        embedding_vector = embeddings_index.get(word);
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector;
    print("Embedding matrix shape: {}".format(embedding_matrix.shape));
    return embeddings_matrix, (len(word_index_training) + 1);

def run_keras_model(model):
    global Y_train;
    global Y_test;
    # prepare data and labels
    X_train_oh, X_test_oh = compute_one_hot_encoding();
    Y_train = [0 if x == "false" else 1 for x in Y_train];
    Y_train_cat = to_categorical(Y_train, num_classes=2); # [1. 0.] ==> [[0. 1.], [1. 0.]]
    Y_test = [0 if x == "false" else 1 for x in Y_test];
    Y_test_cat = to_categorical(Y_test, num_classes=2);
    X_train_oh = np.array(X_train_oh);
    X_test_oh = np.array(X_test_oh);
    # customize optimizer
    if args.optimizer == "sgd":
        sgd = optimizers.SGD(lr=0.001, clipvalue=0.5, momentum=0.0, decay=0.95, nesterov=False);
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy']);
    elif args.optimizer == "adam":
        adam = optimizers.Adam(lr=0.001, decay=0.95);
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy']);
    print(model.summary());
    # compile, fit and predict
    model.fit(X_train_oh, np.array(Y_train), batch_size=int(float(args.batch_size)), epochs=int(float(args.epochs)),
              shuffle=True);  # validation_split=0.2
    Y_pred = model.predict(X_test_oh);
    print("Check predicted classes per data instance:");
    for pred in Y_pred:
        print(pred);
    Y_pred_list = list();
    # transform predicted classes to list that will be fed to the evaluation metric func
    for arr in Y_pred:
        predicted_class_idx = np.argmax(arr);  # 0 for negative class, 1 for positive class
        if predicted_class_idx == 0:  # [1. 0.] ==> [[0. 1.], [1. 0.]]
            Y_pred_list.append(0);
        else:
            Y_pred_list.append(1);
    print("Y_train: {}".format(Y_train));
    print("Y_test: {}".format(Y_test));
    print("Y_pred_list: {}".format(Y_pred_list));
    return Y_pred_list, Y_test, "1"

def set_up_embedding():
    if args.embedding_type == "not-pretrained":
        return Embedding(int(float(args.maximum_word_number)), int(float(args.embedding_dimension)),
                         input_length=int(float(args.maximum_sequence_length)));
    elif "glove" in args.embedding_type:
        embeddings_matrix, input_dimension = load_glove_embeddings(args);
        return Embedding(input_dimension,
                         int(float(args.embedding_dimension)),
                                    weights=[embeddings_matrix],
                                    input_length=int(float(args.maximum_sequence_length)),
                                    trainable=False);

# load data
with open(args.training_filename, "r") as inputf:
    for line in inputf.readlines():
        text = line.split("__label__")[0];
        X_train.append(text);
        label = line.split("__label__")[1].replace("\n", "");
        Y_train.append(label);
with open(args.test_filename, "r") as inputf:
    for line in inputf.readlines():
        text = line.split("__label__")[0];
        X_test.append(text);
        label = line.split("__label__")[1].replace("\n", "");
        Y_test.append(label);
print("X_train size: {}".format(len(X_train)));
print("X_test size: {}".format(len(X_test)));
print("Y_train size: {}".format(len(Y_train)));
print("Y_test size: {}".format(len(Y_test)));

# Random Forest
X_train_tf, X_test_tf = compute_tf_idf();
rf_clf = RandomForestClassifier(n_jobs=2, verbose=1, class_weight="balanced", random_state=0);
rf_clf.fit(X_train_tf, Y_train);
clf_result = rf_clf.predict(X_test_tf);
compute_confusion_matrix(pd.DataFrame(data={"predicted": list(clf_result), "actual": Y_test}), pos_label='true');

# Fasttext
fasttext_clf = fasttext.supervised(args.training_filename, 'model');
clf_result = fasttext_clf.predict(X_test, k=len(X_test));
Y_pred = list();
for result in clf_result:
    Y_pred.append(result[0]);
compute_confusion_matrix(pd.DataFrame(data={"predicted": Y_pred, "actual": Y_test}), pos_label='true');

# Keras models: LSTMs
model = Sequential();
# choose embedding
embedding_layer = set_up_embedding();
model.add(embedding_layer);
# model.add(Dropout(0.2));
# hidden layers
model.add(Dense(10, input_shape=(int(float(args.maximum_sequence_length)),), activation='relu'));
# model.add(Dropout(0.2));
model.add(Dense(256, activation='sigmoid'));
# model.add(Dropout(0.2));
model.add(Dense(2, activation='softmax'));
# output layer
model.add(Flatten());
model.add(Dense(2, activation='sigmoid'));  # alternatively, one output dimension with binary_crossentropy
Y_pred_list, Y_test, pos_label = run_keras_model(model);
compute_confusion_matrix(result=pd.DataFrame(data={"predicted": Y_pred_list, "actual": Y_test}), pos_label=pos_label);

# Fully connected feed forward network
model = Sequential();
model.add(embedding_layer);
# model.add(Dropout(0.2));
# TODO try direction and pooling
model.add(LSTM(32, kernel_initializer='random_uniform',
                bias_initializer='zeros', return_sequences=True,
                dropout=0.2, recurrent_dropout=0.2));
model.add(LSTM(64, kernel_initializer='random_uniform',
                bias_initializer='zeros', return_sequences=True,
                dropout=0.2, recurrent_dropout=0.2));
# kernel_regularizer=regularizers.l2(0.01)
model.add(LSTM(128, kernel_initializer='random_uniform',
                bias_initializer='zeros', return_sequences=True,
                dropout=0.2, recurrent_dropout=0.2));
# model.add(Dropout(0.2));
model.add(Flatten());
model.add(Dense(2, activation='sigmoid'));  # alternatively, one output dimension with binary_crossentropy
Y_pred_list, Y_test, pos_label = run_keras_model(model);
compute_confusion_matrix(result=pd.DataFrame(data={"predicted": Y_pred_list, "actual": Y_test}), pos_label=pos_label);