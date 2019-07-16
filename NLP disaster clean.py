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


#%% Model 2: capturing semantic meaning
import gensim

# Load Google's pre-trained Word2Vec model.
# word2vec is a 2-layer neural net, 
#    that returns a vector for each word in the vocabulary
#    and words similar in meaning have similar vector values
word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)  

#%% For each tweet, calculate the average word2vec score
def get_average_word2vec(tokens_list, vector):
    """
    For each tweet, calculate the average word2vec score (length 300). 
    Ignore words that are not in the word2vec vocabulary
    tokens_list: a tokenized sentence in list format
    vector: a word2vec model
    Output:
        A 300-dim vector that is the average of each word in a sentence
    """
    if (len(tokens_list) < 1):
        return(np.zeros(300))
    # Ignore a word if a word is not in the vocabulary
    vectorized = [vector[word] if word in vector else None for word in tokens_list]
    # Only keep the words in the vocabulary
    vectorized_clean = []
    for i in range(len(vectorized)):
        if vectorized[i] is not None:
            vectorized_clean.append(vectorized[i])
    # calculate the average semantic vector
    averaged = np.mean(vectorized_clean, axis = 0)
    # if the whole sentence contains no usable information, assign to 0
    if np.isnan(averaged).all():
        return(np.zeros(300))
    return(averaged)

def get_word2vec_embeddings(data_set, vector):
    """
    Wrapper function, apply the get_average_word2vec function to all tweets
    """
    embeddings = data_set['tokens'].apply(
            lambda x: get_average_word2vec(x, vector))
    return list(embeddings)
            
#%% same split as bag of words
embeddings = get_word2vec_embeddings(clean_questions, word2vec)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
        embeddings, list_labels, test_size=0.2, random_state=1)
# tweets # 36 and #40 did not map to a vec 
print('#36:', clean_questions['tokens'][36]) #36: ['looooool']
print('#40:', clean_questions['tokens'][40]) #40: ['cooool']
#%% visualize
from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# latent semantic analysis
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        colors = ['orange','blue','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, 
                        c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='orange', label='Irrelevant')
            green_patch = mpatches.Patch(color='blue', label='Disaster')
            plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


fig = plt.figure(figsize=(6, 6))          
plot_LSA(embeddings, list_labels)

#%% Logistic regression based on the word2vec embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score,classification_report

clf2 = LogisticRegression(random_state = 3, 
                          multi_class = 'multinomial',
                          solver = 'newton-cg').fit(
                                  X_train_word2vec, y_train_word2vec)
yhat_test_word2vec = clf2.predict(X_test_word2vec)
yhat_train_word2vec = clf2.predict(X_train_word2vec)
print("Predicted values:", pd.Series(yhat_test_word2vec).value_counts())
print("Real values:", pd.Series(y_test_word2vec).value_counts())

#%% evaluate the logistic classifier
print("training accuracy:", clf2.score(X_train_word2vec, y_train_word2vec))
print("testing accuracy:", clf2.score(X_test_word2vec, y_test_word2vec))
# training accuracy: 0.8152411649591343
# testing accuracy: 0.7988029465930019
# compared to bag of words embedding, 
#   similar testing accuracy but much smaller variance!

mat2 = confusion_matrix(yhat_test_word2vec, y_test_word2vec)
print('Confusion matrix:', mat2)
#[[1082  258    1]
# [ 177  653    1]
# [   0    0    0]]

prec = precision_score(y_test_word2vec, yhat_test_word2vec, average = None, pos_label = 1)
rec = recall_score(y_test_word2vec, yhat_test_word2vec, average = None, pos_label = 1)
print('precision = {}, recall = {}'.format(prec, rec))

print(classification_report(y_test_word2vec, yhat_test_word2vec))

# later can try LIME (Local Interpretable Model-agnostic Explanations)

#%% Model 3: decision tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = 5)
clf3 = tree.fit(X_train, y_train)
y_pred_3 = tree.predict(X_test)
#%% evaluate model 3
from sklearn.metrics import confusion_matrix, precision_score, recall_score
mat3 = confusion_matrix(y_test, y_pred_3)

print('Model 3 training accuracy:', clf3.score(X_train, y_train))
print('Model 3 testing accuracy:', clf3.score(X_test, y_test))
print('Model 3 confusion matrix: \n',mat3)

# Model 3 training accuracy: 0.9850351099343847
# Model 3 testing accuracy: 0.7472375690607734
# There is a bigger variance than logistic regression model,
#    which is the weakness of decision trees
# Python doesn't support pruning
# Next try bagging (random forest)

#%% Model 4: Random Forest
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(tree, n_estimators = 10, max_samples = 0.8)
clf4 = bag.fit(X_train, y_train)
print('Model 4 training accuracy:', clf4.score(X_train, y_train))
print('Model 4 testing accuracy:', clf4.score(X_test, y_test))

#%% look at hyperparameters
# n_estimators
train_acc = []
test_acc = []
for n in range(1, 9):
    bag = BaggingClassifier(tree, n_estimators = 2**n, max_samples = 0.8)
    clf = bag.fit(X_train, y_train)
    tr = clf.score(X_train, y_train)
    ts = clf.score(X_test, y_test)
    train_acc.append(tr)
    test_acc.append(ts)
    print('With {0} estimators, training accuracy is {1} and testing accuracy is {2}'.format(
            2**n, tr, ts))

# plot the training and testing accuracy of random forest
import matplotlib.pyplot as plt
x = 2** np.arange(1, 9)
plt.plot(x, train_acc, color = 'blue', label = 'training accuracy')
plt.plot(x, test_acc, color = 'green', label = 'testing accuracy')
plt.legend()
plt.xlabel('number of samples')
plt.ylabel('training/testing accuracy')
plt.title('Random forest (max_samples = 0.8)')

# looks like the testing acc levels off at 0.77 after # samples reaches ~40

#%% random patches, max_features = n_features/10
# when first set max_features = sqrt(n_features), the testing acc leveled off at 0.58
train_acc = []
test_acc = []
for n in range(1, 10):
    bag = BaggingClassifier(tree, n_estimators = 2**n, max_samples = 0.8,
                        max_features = 1000, random_state = 15)
    clf = bag.fit(X_train, y_train)
    tr = clf.score(X_train, y_train)
    ts = clf.score(X_test, y_test)
    train_acc.append(tr)
    test_acc.append(ts)
    print('With {0} estimators, training accuracy is {1} and testing accuracy is {2}'.format(
            2**n, tr, ts))

#%% plot the training and testing accuracy of random forest
import matplotlib.pyplot as plt
x = 2** np.arange(1, 10)
plt.plot(x, train_acc, color = 'blue', label = 'training accuracy')
plt.plot(x, test_acc, color = 'green', label = 'testing accuracy')
plt.legend()
plt.xlabel('number of samples')
plt.ylabel('training/testing accuracy')
plt.title('Decision tree-random patches (max_samples=0.8, max_feat=1000)')

# much faster than random forest, but more bias
# testing accuracy leveled off at 0.755 at n_samples = 64