import utils
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
import scipy.stats
from sgd_classifier import BasicSGDClassifier
from tf_shallow_neural_classifier import TfShallowNeuralClassifier
import sst
import random
import os
from rnn_classifier import RNNClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
from tree_nn import TreeNN
import vsm
from sklearn.feature_extraction.text import CountVectorizer
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import regularizers
from keras import backend as K
from PorterStemmer import PorterStemmer
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tf_shallow_neural_classifier import TfShallowNeuralClassifier
import utils

stemmer = PorterStemmer()
fileStop = open('english.stop', 'r')
stopList = []
for line in fileStop:
    stopList.append(line[0:len(line)-1])
def createCharacterToGenderMap():
    listOfFemaleNames = []
    femalefile = open('dist.female.first.txt', encoding = "ISO-8859-1")
    for line in femalefile:
        nameData = line.split(' ')
        listOfFemaleNames.append(nameData[0])
    listOfMaleNames = []
    malefile = open("dist.male.first.txt", "r")
    for line in malefile:
        nameData = line.split(' ')
        listOfMaleNames.append(nameData[0])
    
    file = open('movie_characters_metadata.txt', encoding = "ISO-8859-1")
        
    characterToGender = {}
    femaleCount = 0
    maleCount = 0
    unknownCount = 0
    for line in file:
        characterData = line.split('+++$+++ ')
        gender = characterData[4][0]
        charID = characterData[0].strip()
        if (gender is 'f' or gender is 'F'):
            characterToGender[charID] = 'F'
            femaleCount += 1
        elif (gender is 'm' or gender is 'M'):
            characterToGender[charID] = 'M'
            #characterToGender[characterData[0]] = 'M'
            maleCount += 1
        else:
            name = characterData[1].strip()
            if name in listOfFemaleNames:
                characterToGender[charID] = 'F'
                femaleCount+=1
            elif name in listOfMaleNames:
                characterToGender[charID] = 'M'
                maleCount+=1
            else:
                fullName = name.split(' ')
                firstName = fullName[0]
                x = firstName.split('\'')
                withoutApostrophe = x[0]
                if len(fullName) > 1:
                    lastName = fullName[1]
                else:
                    lastName = firstName
                name.strip()
                firstName.strip()
                lastName.strip()
                withoutApostrophe.strip()
                if ("MALE" in name or "GUY" in name or "MR" in name or "SIR" in name or "UNCLE" in name or "BOY" in name or "HERR" in name or "MAN" in name or "FATHER" in name or "GRANDPA" in name or "DAD" in name or "CAPTAIN" in name or "WAITER" in name):
                    characterToGender[charID] = 'M'
                    maleCount += 1
                elif (firstName in listOfMaleNames):
                    characterToGender[charID] = 'M'
                    maleCount += 1
                elif("FEMALE" in name or "GIRL" in name or "WOMAN" in name or "MS" in name or "WIFE" in name or "MRS" in name or "LADY" in name or "SISTER" in name or "GRANDMA" in name or "MISS" in name or "AUNT" in name or "MOM" in name or "MOTHER" in name or "FRAU" in name or "WAITRESS" in name):
                    characterToGender[charID] = 'F'
                    femaleCount += 1
                elif (firstName is "FEMALE"):
                    characterToGender[charID] = 'F'
                    femaleCount += 1
                elif(firstName in listOfFemaleNames):
                    characterToGender[charID] = 'F'
                    femaleCount += 1
                elif (withoutApostrophe in listOfFemaleNames):
                    characterToGender[charID] = 'F'
                    femaleCount += 1
                elif (withoutApostrophe in listOfMaleNames):
                    characterToGender[charID] = 'M'
                    maleCount += 1
                elif (lastName in listOfMaleNames and "DR" not in firstName and "PROF" not in firstName and "SEN" not in firstName and "PRESIDENT" not in firstName and "BOSS" not in firstName and "PRINCIPAL" not in firstName and "JUDGE" not in firstName):
                    characterToGender[charID] = 'M'
                    maleCount += 1
                elif (lastName in listOfFemaleNames and "DR" not in firstName and "PROF" not in firstName and "SEN" not in firstName and "PRESIDENT" not in firstName and "BOSS" not in firstName and "PRINCIPAL" not in firstName and "JUDGE" not in firstName):
                    characterToGender[charID] = 'F'
                    femaleCount += 1
                else:
                    characterToGender[charID] = gender
                    unknownCount += 1


    print("Num Female: ", femaleCount)
    print("Num Male: ", maleCount)
    print("Num ungendered: ", unknownCount)
    print("TOTAL: ", (femaleCount + maleCount + unknownCount))
    return characterToGender
def createLineToWordsMap():
    file = open('movie_lines.txt', encoding = "ISO-8859-1")
    lineToWords = {}
    for line in file:
        data = line.split('+++$+++ ')
        lineID = data[0].strip()
        text = data[4]
        splitBySpace = text.split(' ')
        words = []
        for word in splitBySpace:
            word = word.lower()
            word = word.strip()
            word = word.strip('?.,!\:-\"\'')
            if word not in stopList:
                #word = stemmer.stem(word)
                if len(word.strip()) > 0 and (word not in stopList):
                    wordWithoutPunctuationAndSpaces = word.strip()
                    wordWithoutPunctuationAndSpaces = wordWithoutPunctuationAndSpaces.strip('?.,!\:;-\"\'')
                    #print(wordWithoutPunctuationAndSpaces)
                    #print(word.strip())
                    word = stemmer.stem(wordWithoutPunctuationAndSpaces)
                    #print("Not in stoplist: ", word)
                    words.append(word)
                    #words.append(wordWithoutPunctuationAndSpaces)
                    #words.append(word.strip())
        lineToWords[lineID] = words
    return lineToWords
def createCharacterToLinesMap():
    file1 = open('movie_conversations.txt', encoding = "ISO-8859-1")
    file2 = open('movie_lines.txt', encoding = "ISO-8859-1")

    charToLinesSpokenAtThem = {}
    charsAndLines = {}
    lineToSpokenBy = {}

    # file1 = characters to lines spoken between them
    for line in file1:
        data = line.split('+++$+++ ')
        character1 = data[0]
        character2 = data[1]
        if ((character1, character2) not in charsAndLines):
            list1 = []
            listOfLines = data[3][1:len(data[3])-2]
            lines = listOfLines.split(',') # all the lines spoken between these char1, char2
            for thing in lines:
                thing = thing.strip() #lineID
                withoutApostrophes = thing[1:len(thing)-1] #remove apostrophes
                list1.append(withoutApostrophes)
            charsAndLines[(((character1, character2)))] = list1
        else:
            list2 = charsAndLines[((character1, character2))]
            listOfLines = data[3][1:len(data[3])-2]
            lines = listOfLines.split(',')
            for thing in lines:
                thing = thing.strip()
                withoutApostrophes = thing[1:len(thing)-1]
                list2.append(withoutApostrophes)
            charsAndLines[(((character1, character2)))] = list2

    # file2: line to spokenBy
    for line in file2:
        data = line.split('+++$+++ ')
        dialogue = data[0].strip()
        spokenBy = data[1].strip()
        lineToSpokenBy[dialogue] = spokenBy

    for key in charsAndLines:
        character1 = key[0].strip()
        character2 = key[1].strip()
        listOfLines = charsAndLines[key]
        for line in listOfLines:
            spokenBy = lineToSpokenBy[line.strip()].strip()
            if (spokenBy == character1):
                spokenTo = character2.strip()
            else:
                spokenTo = character1.strip()
            if (spokenTo not in charToLinesSpokenAtThem):
                linelist = []
                linelist.append(line)
                charToLinesSpokenAtThem[spokenTo] = linelist
            else:
                linelist = charToLinesSpokenAtThem[spokenTo]
                linelist.append(line)
                charToLinesSpokenAtThem[spokenTo] = linelist
    #print(len(charToLinesSpokenAtThem))
    
    newCharToLinesSpokenAtThem = {}
    for char in charToLinesSpokenAtThem:
        if len(charToLinesSpokenAtThem[char]) > 9:
            newCharToLinesSpokenAtThem[char] = charToLinesSpokenAtThem[char]
                
    #print (len(newCharToLinesSpokenAtThem))
    return newCharToLinesSpokenAtThem
def createTrain(charToLinesSpokenAtThem, characterToGender, lineToWords):
    allWords = []
    wordCountMap = {} # map from words to how many times they are spoken
    x_train = []
    y_train = []
    # create map from words spoken @ them -> gender
    wordsSpokenAtThemToGender = {}
    for character in charToLinesSpokenAtThem:
        print(character)
        lines = charToLinesSpokenAtThem[character] #list of lines spoken to character
        words = []
        for line in lines:
            wordsFromLine = lineToWords[line] #convert line to actual words
            #print("WORDS:", wordsFromLine)
            for word in wordsFromLine:
                if (word not in stopList):
                    word = stemmer.stem(word)
                    if word not in allWords:
                        word = stemmer.stem(word)
                        allWords.append(word)
                    if word not in wordCountMap:
                        wordCountMap[word] = 1
                    else:
                        wordCountMap[word] += 1
            words.append(lineToWords[line])
        gender = characterToGender[character]
        x_train.append(words)
        y_train.append(gender)
    print(len(allWords))
    #print(wordCountMap)
    mapForCountVectorizer = {}
    count = 0
    for word in wordCountMap:
        if wordCountMap[word] > 10:
            mapForCountVectorizer[word] = count
            print(word, count)
            count+=1
        
    return(x_train, y_train, mapForCountVectorizer, wordCountMap)
        #wordsSpokenAtThemToGender[words] = gender
characterToGender = createCharacterToGenderMap()
charToLinesSpokenAtThem = createCharacterToLinesMap()
lineToWords = createLineToWordsMap()

train = createTrain(charToLinesSpokenAtThem, characterToGender, lineToWords)

words = train[0][0]
x_train = train[0]
y_train = train[1]
wordMap = train[2]
vectors = []
labels = []
testVectors = []
y_test = []
wordsAsBigStrings = [] #array of all the words spoken to a character in string form
vec = CountVectorizer(vocabulary = wordMap)


count = 0 
for i in range(0, len(x_train)):
    words = x_train[i]
    gender = y_train[i]
    if (count < 8):
        #train/val
        if (gender == 'M'):
            labels.append(0)
        elif (gender == 'F'):
            labels.append(1)
        else: 
            continue 
        finalWords = ""
        for item in words:
            for word in item:
                finalWords += " "
                finalWords += word
        empty = []
        empty.append(finalWords)
        wordsAsBigStrings.append(empty)
        data = vec.fit_transform(empty).toarray()
        vectors.append(data)
 
    elif count < 10:
        #test
        if (gender == 'M'):
            y_test.append(0)
        elif (gender == 'F'):
            y_test.append(1)
        else: 
            continue 
        finalWords = ""
        for item in words:
            for word in item:
                finalWords += " "
                finalWords += word
        empty = []
        empty.append(finalWords)
        wordsAsBigStrings.append(empty)
        data = vec.fit_transform(empty).toarray()
        testVectors.append(data)
         
    elif count == 10:
        # reset
        count = 0
    count = count+1
        
        
featureVectors = []
for i in range(0, len(vectors)):
    featureVectors.append(vectors[i][0])
    
x_test = []
for i in range(0, len(testVectors)):
    x_test.append(testVectors[i][0])
    

print(len(featureVectors))
print(len(labels))
print(len(x_test))
print(len(y_test))

X_train = np.array(featureVectors)
y_train = np.array(labels)
num_classes = 2
from keras import regularizers


y_train = keras.utils.to_categorical(y_train, num_classes)

num_records = X_train.shape[0]
num_features = X_train.shape[1]

print("Feature matrix being passed into network has " + str(num_records) + 
      " characters and " + str(num_features) + " features.")

def neural_network_classifier(learning_rate):
    # create the model
    model = Sequential()
    model.add(Dense(750, input_dim = num_features,kernel_regularizer=regularizers.l2(0.01), activation = 'relu')) # hidden layer
    model.add(Dropout(0.4))
    model.add(Dense(250, activation = 'relu')) # second hidden layer
    model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'softmax')) #output layer
    
    # compile model
    sgd = keras.optimizers.SGD(lr = learning_rate)
    adam = keras.optimizers.Adam(lr = learning_rate)
    
    rmsprop = keras.optimizers.RMSprop(lr = learning_rate)
    model.compile(loss = keras.losses.categorical_crossentropy,
                 optimizer = adam, metrics = ['accuracy'])
    
    print(model.summary())
    return model

from keras import regularizers
import matplotlib.pyplot as plt 

#stop epochs when deltas become close together
batch_size = 256
epochs = 100
learning_rate = 5e-3
verb = 2

model = neural_network_classifier(learning_rate)
#model.fit(X_train, y_train)
history = model.fit(X_train, y_train, 
              epochs = epochs, 
              #batch_size = batch_size,          
              validation_split = 0.25,
              verbose = verb,
              shuffle = True)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

sum = 0.0
for val in history.history['acc']:
    sum+=val
avg = sum/epochs
print("Average Training Accuracy: ", avg)


sum = 0.0
for val in history.history['val_acc']:
    sum+=val
avg = sum/epochs
print("Average Validation Accuracy: ", avg)

print("Last Training Accuracy: ", history.history['acc'][99])
print("Last Validation Accuracy: ", history.history['val_acc'][99])


sum = 0.0
for i in range(90,100):
    sum+= history.history['val_acc'][i]
avg = sum / 10.0
print("Average of last 10 validation: ", avg)

x_testNN = np.array(x_test)
predicted = model.predict_classes(x_testNN)
correct = 0
total = len(y_test)
for i in range(0, len(y_test)):
    if (y_test[i] == predicted[i]):
    #if (y_test[i]==pred[i]):
        correct+=1
        
precision = 0 # out of all the females we caught, how many of them were right? 
totalPrecision = 0
recall = 0 # out of all the females, how many did we catch? 

for i in range(0, len(y_test)):
    if (predicted[i]):
        totalPrecision+=1
        if (y_test[i]):
            precision +=1
    if (predicted[i]):
        recall+=1

totalFemales = 0
for i in range(0, len(y_test)):
    if y_test[i]:
        totalFemales+=1
            

precision = precision/totalPrecision
recall = recall / totalFemales
print(precision, recall)

f1 = 2*(precision*recall) / (precision + recall)

print("F1: ", f1)
        
print(correct)
print(total)
print(correct/total)

plt.plot(history.history['val_acc'])
#plt.plot(predicted)
plt.title('model validiation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper left')
plt.show()
