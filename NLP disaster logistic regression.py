# -*- coding: utf-8 -*-
"""
NLP disaster
Created on Thu Jun 27 15:19:43 2019

@author: Ning
"""
#%%
# ! pip install keras
# ! pip install tensorflow
# ! pip install nltk
import keras # a neural network library
import nltk # natural language toolkit
import pandas as pd
import numpy as np
import re # regular expression operations
import codecs # encoder and decoders, did not work for me
import os # miscellaneous operating system interfaces
# setting working directory
os.chdir("D:\\python\\NLP disaster")
#%% sanitizing input

input_file = open("socialmedia_relevant_cols.csv", "r",
                  encoding="utf-8", errors='replace')
output_file = open("socialmedia_relevant_cols_clean.csv", "w")
def sanitize_characters(raw, clean):    
    for line in raw:
        clean.write(line)
    clean.close()
sanitize_characters(input_file, output_file)

#%%
questions = pd.read_csv("socialmedia_relevant_cols_clean.csv")
questions.head()
questions.tail()
questions.describe() 
# look at the labels
questions['class_label'].unique()
questions['choose_one'].unique()
# frequency
print(questions['choose_one'].value_counts())
print(questions['class_label'].value_counts())
pd.crosstab(questions['choose_one'], questions['class_label'])
#%% clean data
# regular expression operations：
# https://docs.python.org/2/library/re.html
# r is a string literal prefix - but that doesn't seem to be what's happening
# '+' Causes the resulting RE to match 1 or more repetitions of the preceding RE
# [] is used to indicate a set of characters.
#     If the first character of the set is '^', 
#     all the characters that are not in the set will be matched
# ^ indicates the beginning of a string
# \S matches any non-whitespace character
def standardize_text(df, text_field):
    """
    Remove punctuation, hyperlink etc
    df: the data frame to be cleaned
    textfield: the name of the variable to be cleaned
    """
    # remove everything that follows http
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    # remove stand-alone http
    df[text_field] = df[text_field].str.replace(r"http", "")
    # remove any mentions
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    # replace anything that is NOT among the following with a space
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    # change stand-alone @ to at
    df[text_field] = df[text_field].str.replace(r"@", "at")
    # all lower case
    df[text_field] = df[text_field].str.lower()
    return df
questions = standardize_text(questions, "text")
# save for future use
questions.to_csv("clean_data.csv")
questions.head()

clean_questions = pd.read_csv("clean_data.csv")
clean_questions.tail()

# EDA
clean_questions.groupby('class_label').count()

#%% prepare the data

# tokenizing sentences to a list of separate words
from nltk.tokenize import RegexpTokenizer
# \w indicates any alphanumeric character, equivalent to [A-Za-z0-9]
# \w+ is essentially anything that looks like a word
tokenizer = RegexpTokenizer(r'\w+')
# add a new variable to the data frame
clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)

#%% explore the words
all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
print("{0} words total".format(len(all_words)))
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
print('first 20 sentence lengths:',sentence_lengths[0:20])
print("max sentence length is {0}, min sentence length is {1}".format(
        max(sentence_lengths), min(sentence_lengths)))
# some of these words are somehow dropped during the bag of words embedding
#%% see the most frequent words
count = [all_words.count(word) for word in vocab]
word_count = pd.DataFrame({'word':vocab, 'count':count})
word_count.head()

max(count)
#sorted(count, reverse = True )[0:20]
ind = np.greater_equal(count, 500)
word_count.iloc[ind,:]
#%% plot a histogram of the sentence lengths
import matplotlib.pyplot as plt
plt.hist(sentence_lengths)
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')

#%% bag of words embedding
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

list_corpus = clean_questions["text"].tolist()
list_labels = clean_questions['class_label'].tolist()
count_vectorizer = CountVectorizer()

X_counts = count_vectorizer.fit_transform(list_corpus)
vocab = count_vectorizer.get_feature_names()
print("vocab size:", len(vocab))
# train test split
X_train, X_test, y_train, y_test = train_test_split(
        X_counts, list_labels, test_size = 0.2, random_state=1)

print('Training set labels:', pd.Series(y_train).value_counts())
print('Testing set labels:', pd.Series(y_test).value_counts())
#%% explore the bag of words
real_pct = np.mean(X_train[np.equal(y_train,1),:], axis = 0)
faux_pct = np.mean(X_train[np.equal(y_train,0),:], axis = 0)
diff_pct = np.subtract(real_pct, faux_pct)
print('max:', np.max(diff_pct))
print('min:', np.min(diff_pct))
print('top 5%:', np.quantile(diff_pct, 0.95))
print('bottom 5%:', np.quantile(diff_pct, 0.05))
top1pct = np.quantile(diff_pct, 0.99)
bottom1pct = np.quantile(diff_pct, 0.01)
print('top 1%:', top1pct)
print('bottom 1%:', bottom1pct)
# words that are more frequent in relevant tweets
disaster_words = [vocab[i] for i in range(len(vocab)) if diff_pct[0,i]>top1pct]
print('\n disaster words:\n', disaster_words,'\n')
# words that are more frequent in not relevant tweets
faux_words = [vocab[i] for i in range(len(vocab)) if diff_pct[0,i]<bottom1pct]
print('faux words:\n',faux_words)


#%% Using logistic regresion to fit a classifier
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state = 3, 
                          multi_class = 'multinomial',
                          solver = 'newton-cg').fit(X_train, y_train)
yhat_test = clf1.predict(X_test)
yhat_train = clf1.predict(X_train)
print("Predicted values:",pd.Series(yhat_test).value_counts())
print("Real values:",pd.Series(y_test).value_counts())
#%% evaluate the logistic classifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score,classification_report
print("training accuracy:", clf1.score(X_train, y_train))
print("testing accuracy:", clf1.score(X_test, y_test))
#training accuracy: 0.9651202946932198
#testing accuracy: 0.7969613259668509

mat = confusion_matrix(yhat_test, y_test)
print(mat)
prec = precision_score(y_test, yhat_test, average = None, pos_label = 1)
rec = recall_score(y_test, yhat_test, average = None, pos_label = 1)
print('precision = {}, recall = {}'.format(prec, rec))
print(classification_report(y_test, yhat_test))

# it seems like some true disaster tweets are not picked up. 

#%% get most important features
# strategy: pair the coefficients with the features, 
# then sort the coefficients and get the corresponding features
# note: lambda is an anonymous function. 
#     It allows you to define a fn on the spot without having to give it a name
def get_important_features(model, vocab, n=5):
    """
    Sort the coefficients of a model and get the corresponding features.
    model: for now, a logistic regression model
    vocab: the vocabulary associated with the model
    n: the number of top features to extract
    """
    # loop for each class
    classes = {}
    for k in range(model.coef_.shape[0]):
        importance = {vocab[i]:clf1.coef_[k][i] for i in range(len(vocab))}
        importance_sorted = sorted(importance.items(), 
                                   key = lambda kv:(kv[1], kv[0]),
                                   reverse = True)
        top = importance_sorted[:n]
        bottom = importance_sorted[-n:]
        classes[k] = {
                "top": top,
                "bottom": bottom}
    return classes

features1 = get_important_features(clf1, vocab, n = 10)

#%% plot the most important features using a horizontal bar plot
fig = plt.figure(figsize = (15, 5))
fig.subplots_adjust(wspace = 0.4)
features = features1
for k in range(3):
    plt.subplot(1,3,k+1)
    n = len(features[k]["top"])
    y_pos1 = np.arange(1,n+1)[::-1]
    y_pos2 = -np.arange(1,n+1)
    width1 = [pair[1] for pair in features[k]["top"]]
    width2 = [pair[1] for pair in features[k]["bottom"]]
    label1 = [pair[0] for pair in features[k]["top"]]
    label2 = [pair[0] for pair in features[k]["bottom"]]
    plt.barh(y = np.concatenate((y_pos1, y_pos2)), 
         width = np.concatenate((width1, width2)))
    plt.yticks(ticks = np.concatenate((y_pos1, y_pos2)),
           labels = np.concatenate((label1, label2)))
    plt.xlabel('Coefficient')
    plt.ylabel('Important Words')
    plt.title(['Irrelavant','Disaster','Can\'t Decide'][k])


