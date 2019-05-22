# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 12:50:41 2019

@author: zxc63
"""

import sklearn
import nltk
from sklearn.externals import joblib
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
def read_iob2_sents(in_iob2_file):
    sents= []
    with open(in_iob2_file, 'r', encoding = 'utf8') as f:
        for sent_iob2 in f.read().split('\n\n'):
            sent = []
            for raw in sent_iob2.split('\n'):
                if raw == '':
                    continue
                columns = raw.split('\t')
                sent.append(tuple(columns))
            sents.append(sent)
    return sents
def sent2labels(sent):
    return [label for token, label in sent]
def sent2tokens(sent):
    return [token for token, label in sent]
def word2features(sent, i): 
    word = sent[i][0]
    features = {
            'word': word, 
            'word.islower()': word.islower(),
            'word.isalnum()': word.isalnum(),
            'word.isupper()': word.isupper(),
            'word.isdigit()': word.isdigit(),
            'word.isalpha()': word.isalpha()}
    if i> 0:
        word1 = sent[i-1][0]
        features.update({
             '-1:word': word1,
                '-1:word.islower()': word1.islower(),
                '-1:word.isalnum()': word1.isalnum(), 
                '-1:word.isupper()': word1.isupper(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.isalpha()': word1.isalpha()})
    else:
        features['BOS'] = True
    if i< len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
                '+1:word': word1,
                '+1:word.islower()': word1.islower(),
                '+1:word.isalnum()': word1.isalnum(), 
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.isalpha()': word1.isalpha()})
    else:
        features['EOS'] = True
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def write_iob2(data, pred, out_iob2_file):
    with open(out_iob2_file, 'wb') as iob2_writer:
        for _data, _pred in zip(data, pred):
            for _tuple, _pred in zip(_data, _pred):
                iob2_writer.write(bytes('\t'.join(_tuple) + '\t' + _pred+ '\n', encoding = 'utf8'))
            iob2_writer.write(bytes('\n', encoding = 'utf8'))
        print(iob2_writer)
 
"""       
def posTagging(word):
    word = nltk.word_tokenize(word)
    word=nltk.pos_tag(word)
    return word[0][1]
    """
train_sents= read_iob2_sents('sample_data.iob2')+read_iob2_sents('sample_data2.iob2')

X = [sent2features(s) for s in train_sents]
y = [sent2labels(s) for s in train_sents]

crf= sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        min_freq=0)
crf.fit(X, y)
joblib.dump(crf, 'crf2.pkl')
del X
del y
test_sents= read_iob2_sents('test_data.iob2')
X_test= [sent2features(s) for s in test_sents]
y_test= [sent2labels(s) for s in test_sents]
y_pred= crf.predict(X_test)
y_pred
labels = list(crf.classes_)
labels.remove('O')
sorted_labels= sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
write_iob2(test_sents, y_pred, 'pred_sample_data2.iob2')