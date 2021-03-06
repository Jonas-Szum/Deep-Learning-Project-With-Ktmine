from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras import regularizers
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import *
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Conv2D, GlobalMaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
import re
import random
import xlrd
import codecs
import os.path
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from os import path
from keras.constraints import max_norm
from nltk.tokenize import sent_tokenize, word_tokenize
norm = max_norm(3.0)
sampleSet = xlrd.open_workbook('./paragraphID/agreements.xlsx') #parse the sample set as CSV
eachRow = sampleSet.sheet_by_index(0) #parse each row as an individual set of classifications by ID

docID1 = []
paraNum1 = []
paragraphs1 = []
payments1 = []
rr1 = []

for rowIndex in range(1, eachRow.nrows): #ignore first row
  i = 0 #i represents the column
  row = eachRow.row(rowIndex)
  for idx, row_obj in enumerate(row):
    if i == 0:
      docID1.append(int(row_obj.value))
    elif i == 1:
      paraNum1.append(int(row_obj.value))
    elif i == 2:
      paragraphs1.append(str(row_obj.value))
    elif i == 3:
      payments1.append(str(row_obj.value))
    elif i == 4:
      rr1.append(str(row_obj.value))
    i += 1

docID = []
paraNum = []
paragraphs = []
payments = []
rr = []
#remove blank labels from the training set
for i in range(len(docID1)): #could be any of the above 5 lists
  if ((rr1[i] != 'Y' and rr1[i] != 'N') or (payments1[i] != 'Y' and payments1[i] != 'N')): #if any classifications are blank, skip
    continue
  else:
    docID.append(docID1[i])
    paraNum.append(paraNum1[i])
    paragraphs.append(paragraphs1[i])
    payments.append(payments1[i])
    rr.append(rr1[i])

#trainCorpus = []
trainCorpus = paragraphs

maxFeat = 5000 #maxFeat is the number of allowed words to be used as features

myTfIdf = TfidfVectorizer(stop_words = 'english', ngram_range = (1, 3), max_features = maxFeat)
transformedIDF = myTfIdf.fit_transform(trainCorpus)

rrBin = [] #binary rr classification for corresponding documents
paymentsBin = [] #binary payment classification for corresponding documents
#rrBin and paymentsBin will be trained seperately

for i in range(len(rr)): #could be replaced with len(payments)
  if rr[i] == 'N': #if not an rr, add 0 to the list
    rrBin.append(0)
  else:       #some data is missing, what to do with cell that aren't N or Y?
    rrBin.append(1)

  if payments[i] == 'N': #if not a payment, add 0 to the list
    paymentsBin.append(0)
  else:
    paymentsBin.append(1)

print('rr len: ', len(rrBin))
print('payments len: ', len(paymentsBin))
print('corpus len: ', len(trainCorpus))
"""
for index in topFiveIndices: #index represents each possible classification
  specificClassifier = [] #specificClassifier represents each file's relation to the current classification (index) being checked.
  for fileName in trainNames:
    if fileName in filesPerTagIndex[index]: #if the file is listed in this classification's table, append a 1 to the list. the position in this list will correspond to fileName's position in the trainName list
      specificClassifier.append(1)
    else:
      specificClassifier.append(0)
  topClassifiers.append(specificClassifier) #for each index, the file names are classified as 1 or 0 depending on the document type
"""
model = MLPClassifier(hidden_layer_sizes = (4, 25), max_iter=1000, solver='adam')
print('rr: ', np.mean(cross_val_score(model, transformedIDF, rrBin, cv=10)))
rrTrain, rrTest, outputRRTrain, outputRRTest = train_test_split(transformedIDF, rrBin, train_size=.9)
model.fit(rrTrain, outputRRTrain)
predictedVals = model.predict(rrTest)
print('Confusion matrix for the rr (false negatives is 1,0 and false positives is 0,1)')
print(confusion_matrix(predictedVals, outputRRTest)) #compare our predicted vals with the actual test labels

model = MLPClassifier(hidden_layer_sizes = (4, 25), max_iter=10000, solver='adam')
print('payment: ', np.mean(cross_val_score(model, transformedIDF, paymentsBin, cv=10)))
paymentsTrain, paymentsTest, outputPaymentsTrain, outputPaymentsTest = train_test_split(transformedIDF, paymentsBin, train_size=.9)
model.fit(paymentsTrain, outputPaymentsTrain)
predictedVals = model.predict(paymentsTest)
print('Confusion matrix for the payments (false negatives is 1,0 and false positives is 0,1)')
print(confusion_matrix(predictedVals, outputPaymentsTest)) #compare our predicted vals with the actual test labels

