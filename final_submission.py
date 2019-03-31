import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import IsolationForest
import csv

data = pd.read_csv("train_file.csv")
input_len = data.shape[0]

labels = data['MaterialType'].values.flatten()	# make a list from the vertical column
data = data.drop(['ID','CheckoutMonth','CheckoutYear','Checkouts','PublicationYear','MaterialType'], axis = 1)
train_text = data['CheckoutType'].map(str)+' '+data['UsageClass'].map(str)+' '+data['Title'].map(str)+' '+data['Creator'].map(str)+' '+data['Subjects'].map(str)+' '+data['Publisher'].map(str)
del data

le = preprocessing.LabelEncoder()
le.fit(labels)
train_labels = le.transform(labels)
del labels


data = pd.read_csv("test_file.csv")

print(data.head())
data = data.drop(['ID','CheckoutMonth','CheckoutYear','Checkouts','PublicationYear'], axis = 1)
test_text = data['CheckoutType'].map(str)+' '+data['UsageClass'].map(str)+' '+data['Title'].map(str)+' '+data['Creator'].map(str)+' '+data['Subjects'].map(str)+' '+data['Publisher'].map(str)
del data

all_text = train_text
all_text = all_text.append(test_text)

sw = stopwords.words('english')
ss = SnowballStemmer('english')

corpus = []			# preproccessing both train and text data
for row in all_text:
    row = row.lower()
    row = row.split()	# splitting each string to words
    row = [ss.stem(word) for word in row if not word in set(sw)]
    row = ' '.join(row)	# joining back the words to form string
    corpus.append(row)	# the entire collection is named corpus
print(np.shape(corpus));
vectorizer = CountVectorizer().fit(corpus)		
train_features = vectorizer.transform(corpus)	#keeping count of each word in each string -tuple of pair and int


forest = IsolationForest(n_estimators=100, contamination=0.045)
forest.fit(train_features)
outliers = forest.predict(train_features)
# print(outliers.shape)

new_corpus=[]
for i in range(0,len(outliers)):
    if outliers[i]==1:			# +ve score means inliers
        new_corpus.append(corpus[i])	#in lists append is inplace

out = []
for i in range(0,input_len):
    if outliers[i]==1:			# +ve score means inliers
        out.append(i)
# print(np.shape(out))

corpus_in = []
train_labels_in = []

for i in out:
    corpus_in.append(corpus[i])
    train_labels_in.append(train_labels[i])

#print(np.shape(out))       
#print(np.shape(train_labels_in))
#print(np.shape(corpus_in))
#print(train_labels.shape)
vectorizer2 = CountVectorizer().fit(new_corpus)
train_features_in = vectorizer2.transform(corpus_in)
test_features = vectorizer2.transform(test_text)


# print(np.shape(train_features_in))
# X_train, X_test, Y_train, Y_test = train_test_split(train_features_in, train_labels_in, test_size = 0.2, random_state = 23)

clf = XGBClassifier(learning_rate=0.965)
# clf.fit(X_train,Y_train)
clf.fit(train_features_in, train_labels_in)

pred = clf.predict(test_features)
# pred = clf.predict(X_test)
# acc = accuracy_score(pred, Y_test)
# print(acc) 

writer = csv.writer(open("result.csv", "w"))
head = ["ID", "MaterialType"]
writer.writerows([head])
cnt = 31654
for row in pred:
    writer.writerows([[str(cnt), str(le.inverse_transform(row))]])
    cnt = cnt+1

