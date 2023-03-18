import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes_list = []
documents_list = []
words_to_ignore = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents_list.append((w, intent['tag']))

        if intent['tag'] not in classes_list:
            classes_list.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in words_to_ignore]
words = sorted(list(set(words)))
classes_list = sorted(list(set(classes_list)))
print(len(documents_list), "documents")
print(len(classes_list), "classes", classes_list)
print(len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes_list,open('classes.pkl','wb'))

training_words_list = []
empty_output = [0] * len(classes_list)
for doc in documents_list:
    bag = []
    words_with_pattern = doc[0]
    words_with_pattern = [lemmatizer.lemmatize(word.lower()) for word in words_with_pattern]
    for w in words:
        bag.append(1) if w in words_with_pattern else bag.append(0)
    
    row_of_output = list(empty_output)
    row_of_output[classes_list.index(doc[1])] = 1
    
    training_words_list.append([bag, row_of_output])
random.shuffle(training_words_list)
training_words_list = np.array(training_words_list)
train_x = list(training_words_list[:,0])
train_y = list(training_words_list[:,1])
print("Training data created")

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created....")