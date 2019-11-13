# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:18:14 2019

@author: Ning
"""
import os
import pandas as pd
import numpy as np
os.chdir("D:\\python\\NLP disaster")
clean_questions = pd.read_csv("clean_data.csv")
clean_questions.groupby('class_label').count()

#%% import word2vec
import gensim
word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)  

#%% tokenize into integers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_questions["text"])
sequences = tokenizer.texts_to_sequences(clean_questions["text"])
# this tokenizer does not remove 's 


#%% prepare data for CNN
from sklearn.model_selection import train_test_split
max_squence_length = 34
cnn_data = pad_sequences(sequences, maxlen = max_squence_length)
print(cnn_data.shape)
cnn_labels = to_categorical(clean_questions["class_label"])

train_x, test_x, train_y, test_y = train_test_split(
        cnn_data, 
        cnn_labels, 
        test_size = 0.2)
print('training shape: x: {0}, y: {1}'.format(
        train_x.shape, train_y.shape))
print('testing shape: x: {0}, y: {1}'.format(
        test_x.shape, test_y.shape))

#%% create embedding matrices
# mapping between word and integer
embedding_dim = 300
word_index = tokenizer.word_index
embedding_weights = np.zeros((len(word_index) + 1, embedding_dim))
for word, index in word_index.items():
     embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(embedding_dim)

embedding_weights.shape

#%% define the NN
from keras.layers import Dense, Input, Flatten, Dropout, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

def ConvNet(embeddings, max_sequence_length, embedding_dim, trainable = False,
            num_categories = 3):
    """
    embeddings: word2vec embedding matrix, shape (vocab_size, embedding_dim)
    max_sequence_length: maximum sentence length in each document. default 34
    embedding_dim: currently 300
    trainable: whether the embedding layer is trainable. 
        Because its pretrained, the default is set to False
    num_categories: number of categories in the label. in this case it's 3
    """
    sequence_input = Input(shape = (max_sequence_length, ))
    
    embedding_layer = Embedding(input_dim = len(word_index) + 1,
                                output_dim =embedding_dim,
                                weights = [embeddings], # use when vec is pretrained
                                input_length = max_sequence_length,
                                trainable = trainable)
    
    embedded_sequences = embedding_layer(sequence_input)
    
    # Yoon Kim structure, training 3, 4, 5-grams together
    convs = []
    filter_sizes = [3, 4, 5]
    for filter_size in filter_sizes:
        conv_layer = Conv1D(filters = 100, 
                            kernel_size = filter_size,
                            activation = 'relu')(embedded_sequences)
        pool_layer = MaxPooling1D(pool_size = 3)(conv_layer)
        convs.append(pool_layer)
    l1 = Concatenate(axis = 1)(convs)
    x = Dropout(0.5)(l1)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    
    preds = Dense(num_categories, activation = 'softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(optimizer = 'adam', 
                  loss = 'categorical_crossentropy', 
                  metrics = ['acc'])
    
    return model

#%% run the model
model = ConvNet(embeddings = embedding_weights,
        max_sequence_length = 34,
        embedding_dim = embedding_dim
        )
model.fit(x = train_x, y = train_y,
          validation_data = (test_x, test_y),
          batch_size = 128, epochs = 3)

