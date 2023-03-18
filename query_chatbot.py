import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def preproc_sentence(sent):
    word_tokens = nltk.word_tokenize(sent)
    word_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokens]
    return word_tokens

def bag_of_words(sent, words, show_details=True):
    word_tokens = preproc_sentence(sent)
    bag = [0]*len(words)  
    for s in word_tokens:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def class_prediction(sent, model):
    p = bag_of_words(sent, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    reply_results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    reply_results.sort(key=lambda x: x[1], reverse=True)
    reply_list = []
    for r in reply_results:
        reply_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return reply_list

def getReply(msg,ints, intents_json):
    tag = ints[0]['intent']
    intents_list = intents_json['intents']
    for i in intents_list:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    greet= ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day","Hii","how you doing","hello","hi","Hii","Hi","hey","hola","hi there"]
    if(tag=='greeting' and (msg not in greet)):
        result = random.choice(["Sorry, can't understand you", "Please give me more info", "Not sure I understand"])
    return result

def chatbot_reply_response(message):
    nums = class_prediction(message, model)
    result = getReply(message,nums, intents)
    return result


#Creating GUI with tkinter
import tkinter
from tkinter import *

def send():
    message = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if message != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + message + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_reply_response(message)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("IITBhilai_Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

C = Canvas(base, bg="#B2B2B2", height=395, width=400)
filename = PhotoImage(file = "abi95-mz7nz-001.png")
base.iconphoto(False, filename)
#Create Chat window
ChatLog = Text(base, bd=0, font="Verdana", bg="white", height="8", width="50", fg="#00ABB3",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#00ABB3", activebackground="#54B435",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial",fg="#54B435")
#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=271, y=401, height=90)


C.pack()

base.mainloop()
