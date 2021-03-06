from sklearn.model_selection import cross_val_score
from keras import regularizers
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from nltk.stem import WordNetLemmatizer
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
sampleSet = xlrd.open_workbook('./Data/dataset.xlsx') #parse the sample set as CSV
eachRow = sampleSet.sheet_by_index(0) #parse each row as an individual set of classifications by ID


#The following two functions have been borrowed from https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.Xkmm2ulMG8g
def sort_coo(coo_matrix):
  tuples = zip(coo_matrix.col, coo_matrix.data)
  return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
  """get the feature names and tf-idf score of top n items"""

  #use only topn items from vector
  sorted_items = sorted_items[:topn]

  score_vals = []
  feature_vals = []

  # word index and corresponding tf-idf score
  for idx, score in sorted_items:
    #keep track of feature name and its corresponding score
    score_vals.append(round(score, 3))
    feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
  results= {}
  for idx in range(len(feature_vals)):
    results[feature_vals[idx]]=score_vals[idx]

  return results



correctCount = 0
totalCount = 0
tagCounts = {}  #used for human understanding
tagIndices = {} #used to index the file for the ML model
fileNames = []
fileNumbers = []
#tagsAssociated will be used to keep track of the tags associated with a specific file name
tagsAssociated = {}
tagsAssociatedHuman = {}
trainNumbers = {}

for rowIndex in range(1, eachRow.nrows): #ignore first row
  i = 0 #i represents the column
  row = eachRow.row(rowIndex)
#  print('')
  for idx, row_obj in enumerate(row): #separate useless idx from row_obj, where row_obj is a 1 or a 0
    if(i >= 5 and i <= 81 and row_obj.value == 1.0): #identification begins at index 5 and ends at index 81
#      print('{}: {}'.format(eachRow.row(0)[i].value, row_obj.value)) #(0)[i].value obtains the row of names, (0), i obtains the column of the name we're selecting. row_obj.value is a 1 or 0
      if eachRow.row(0)[i].value in tagCounts: #if the label is in the dictionary, increment it so we can count num occurences
        tagCounts[eachRow.row(0)[i].value] += 1
        tagIndices[i] += 1
      else:
        tagCounts[eachRow.row(0)[i].value] = 1 #if it's the first occurence, create it in the dictionary
        tagIndices[i] = 1

      tagsAssociated[fileNames[len(fileNames)-1]].append(i) #if the column i is a tag of the current file name, add it to its associated list
      tagsAssociatedHuman[fileNames[len(fileNames)-1]].append(eachRow.row(0)[i].value)
    elif(i == 0):
      tagsAssociatedHuman['./Data/data/{}.htm'.format(int(row_obj.value))] = [] #human readable version to check accuracy
      tagsAssociated['./Data/data/{}.htm'.format(int(row_obj.value))] = [] #tags associated with the file name starts at 0
      trainNumbers['./Data/data/{}.htm'.format(int(row_obj.value))] = (int(row_obj.value)) #save just the number for when creating a file
      if(path.exists('./Data/data/{}.htm'.format(int(row_obj.value))) == True): #check to see if the file exists
        fileNames.append('./Data/data/{}.htm'.format(int(row_obj.value)))
        #fileNumbers.append(int(row_obj.value))
      else:
        print('File {}.htm doesn\'t exist'.format(int(row_obj.value)))
        break
    i += 1
#  print('')

topFiveIndices = []
for index in tagIndices:
  if tagIndices[index] >= 50:
    topFiveIndices.append(index)

topFiveTags = sorted(dict(Counter(tagCounts).most_common(5)).keys()) #human readable version of top 5 labels

topFiveIndices.sort(key = lambda x:tagIndices[x], reverse = True) #sort indices in descending order
print(topFiveIndices)
for index in topFiveIndices:
  print(tagIndices[index])
#print(tagIndices[5]) #most common tag
#print(topFiveTags)
#print(topFiveIndices)
#print('')
#print(tagsAssociated)
#for j in tagsAssociatedHuman:
#  print('{}: {}'.format(j, tagsAssociatedHuman[j]))

random.seed(5000)
random.shuffle(fileNames)
wordSplit = len(fileNames)
trainNames = fileNames[:wordSplit]
#testNames = fileNames[wordSplit:]

trainCorpusBeta = []
testCorpusBeta = []


wordnet = WordNetLemmatizer()

for name in range(0, len(trainNames)):
  myFile = open(trainNames[name])
  txtVersion = myFile.read()
  txtVersion = txtVersion.lower()
  txtVersion = re.sub("<!--?._*?-->","",txtVersion) #experimental, may delete if low accuracy
  txtVersion = re.sub("(\\d|\\W)+"," ",txtVersion)
  tokens = nltk.word_tokenize(txtVersion)
  stem_sentence = []
  for word in tokens:
    lemmatized = wordnet.lemmatize(str(word))
    if(len(lemmatized) > 3):
      stem_sentence.append(lemmatized)
  txtVersion = " ".join(stem_sentence)
  newFile = open('./parsedData/{}.txt'.format(trainNumbers[trainNames[name]]), "w")
  newFile.write(txtVersion)
  newFile.close()
#  trainCorpusBeta.append(txtVersion) #we now have all the texts corresponding to the filename, ready to extract into features



#now, save every lemmatized corpus as a new txt file


