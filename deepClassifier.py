from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pyodbc
#from sklearn.metrics import *
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import tensorflow as tf
import torch
import re
import random
import xlrd
import codecs
import pandas as pd
import os.path
import nltk
#uncomment the lines below for the first time running the program
#nltk.download('punkt')
#nltk.download('wordnet')
from os import path
from keras.constraints import max_norm
from nltk.tokenize import sent_tokenize, word_tokenize


norm = max_norm(3.0)
sampleSet = xlrd.open_workbook('./Data/dataset.xlsx') #parse the sample set as CSV
eachRow = sampleSet.sheet_by_index(0) #parse each row as an individual set of classifications by ID


#The 87th column (index 86) of each row is the name of the file with extension, but doesnt have corresponding files in the fileset?
#Use column 1 (index 0) as the name
#The second, third, and fourth column of each row is useless
#The fifth row counts the number of types the current document is; useless unless we're optimizing performance

#The sixth row (index 5) starts the binary classification of each file
#The eighty-second row (index 81) ends the binary classification of each file
#The total length of binary classifications is 77

correctCount = 0
totalCount = 0
tagCounts = {}  #used for human understanding
tagIndices = {} #used to index the file for the ML model
fileNames = []

#tagsAssociated will be used to keep track of the tags associated with a specific file name
tagsAssociated = {}
tagsAssociatedHuman = {}
for rowIndex in range(1, eachRow.nrows): #ignore first row
  i = 0 #i represents the column
  row = eachRow.row(rowIndex)
  for idx, row_obj in enumerate(row): #separate useless idx from row_obj, where row_obj is a 1 or a 0
    if(i >= 5 and i <= 81 and row_obj.value == 1.0): #identification begins at index 5 and ends at index 81
      if eachRow.row(0)[i].value in tagCounts: #if the label is in the dictionary, increment it so we can count num occurences
        tagCounts[eachRow.row(0)[i].value] += 1
        tagIndices[i] += 1
      else:
        tagCounts[eachRow.row(0)[i].value] = 1 #if it's the first occurence, create it in the dictionary
        tagIndices[i] = 1

      tagsAssociated[fileNames[len(fileNames)-1]].append(i) #if the column i is a tag of the current file name, add it to its associated list
      tagsAssociatedHuman[fileNames[len(fileNames)-1]].append(eachRow.row(0)[i].value)
    elif(i == 0):
      tagsAssociatedHuman['./parsedData/{}.txt'.format(int(row_obj.value))] = [] #human readable version to check accuracy
      tagsAssociated['./parsedData/{}.txt'.format(int(row_obj.value))] = [] #tags associated with the file name starts at 0
      if(path.exists('./parsedData/{}.txt'.format(int(row_obj.value))) == True): #check to see if the file exists
        fileNames.append('./parsedData/{}.txt'.format(int(row_obj.value)))
      else:
#        print('File {}.txt doesn\'t exist'.format(int(row_obj.value)))
        break
    i += 1

topIndices = []
for index in tagIndices:
  if tagIndices[index] >= 50:
    topIndices.append(index)

topIndices.sort(key = lambda x:tagIndices[x], reverse = True) #sort indices in descending order


random.seed(5000)
random.shuffle(fileNames)
wordSplit = len(fileNames)
trainNames = fileNames[:wordSplit] #trainNames is the entire given dataset

trainCorpus = []
for name in range(0, len(trainNames)):
  myFile = open(trainNames[name])
  txtVersion = myFile.read()
  trainCorpus.append(txtVersion) #we now have all the texts corresponding to the filename, ready to extract into features


maxFeat = 5000 #maxFeat is the number of separate words to be saved accross all documents. Tune this accordingly
tokenizer = Tokenizer(num_words = maxFeat, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(trainCorpus) #will be sequenced later

docLens = []
for doc in trainCorpus:
  docLens.append(len(doc))
docMean = int(np.mean(docLens)) #Very large number, only recommended for powerful computers

#####
docMean = 50000 #reasonable vocab size, runs well on computers with no GPU
#####

#print('Average doc length: ', docMean)
sequencedDocs = tokenizer.texts_to_sequences(trainCorpus)
sequencedDocs = pad_sequences(sequencedDocs, maxlen = docMean)
#print('shape of data Tensor: ', sequencedDocs.shape)
#At this point, might need to shave down every document to a certain size



#isTopFive = {} #for each document in the traning set, labeled by name, there will be an array of size 5, indicating whether each document can be classified as any/some/all of the top 5 document types
filesPerTagIndex = {}

topClassifiers = []
for fileName in trainNames:
  singleClass = []
  for index in topIndices:
    if index in tagsAssociated[fileName]:
      singleClass.append(1)
    else:
      singleClass.append(0)
  topClassifiers.append(singleClass) #for all 998 names, each valid classification will be labeled as 1 or 0

df = pd.DataFrame(np.array(topClassifiers), columns=topIndices) #use a dataframe, which is usable for training on keras models
sequencedClassifications = pd.get_dummies(df)
X_train, X_test, Y_train, Y_test = train_test_split(sequencedDocs, sequencedClassifications, test_size = 0.10, random_state = 42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #if you're blesses with a GPU, use it

embed = 64 #change these parameters to mess with the accuracy/runtime trade off
filters = 250
kernel_sz = 3

model = Sequential()
model.add(Embedding(maxFeat, embed, input_length=docMean))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_sz, padding='valid', activation='relu', strides = 1))
model.add(GlobalMaxPooling1D())
model.add(Dense(len(topIndices), activation='sigmoid')) #len(topIndices) indicates how many predictions the model has to make for each document
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 5
batch_size = 2
#print(model.summary()) #Can comment this out if you dont want to see model summary during runtime
history = model.fit(X_train, Y_train, epochs=epochs,
                    batch_size=batch_size, validation_data=(X_test, Y_test))


predictionRate = 0.1 #The lower this is, the lower the false-negative rate.
#### Below is non-essential to the program, only for testing and tuning purposes ###
Y_pred = model.predict(X_test) #Y_pred is our guess, actual labels are Y_test
falseNeg = 0
falsePos = 0
trueNeg = 0
truePos = 0

total = 0
Y_test = Y_test.to_numpy()
for i in range(len(Y_pred)):
  for j in range(len(Y_pred[i])):
    if(Y_pred[i][j] < predictionRate and Y_test[i][j] == 1):
      falseNeg += 1
    elif(Y_pred[i][j] >= predictionRate and Y_test[i][j] == 0):
      falsePos += 1
    elif(Y_pred[i][j] < predictionRate and Y_test[i][j] == 0):
      trueNeg += 1
    elif(Y_pred[i][j] >= predictionRate and Y_test[i][j] == 1):
      truePos += 1
    else:
      print("Y_pred: {}, Y_Test: {}".format(Y_pred[i][j], Y_test[i][j]))
    total += 1


print("Total number of iterations tested: ", total)
print("True positives: ", truePos)
print("True negatives: ", trueNeg)
print("False positives: ", falsePos)
print("False negatives: ", falseNeg)
print('')
falseNegRate = float(falseNeg)/float(total)
falsePosRate = float(falsePos)/float(total)
print("False negative rate: ", falseNegRate)
print("False positive rate: ", falsePosRate)


#server = [REDACTED]
#database = [REDACTED]
#username = [REDACTED]
#password = [REDACTED]
#table_name = [REDACTED]

#connectString = 'Driver={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password+';'

#print(connectString)
#conn = pyodbc.connect(connectString)
#print('connected')
#cursor = conn.cursor()

#cursor.execute('SELECT * FROM '+ database + "." + table_name)
#for row in cursor:
#  print(row)
