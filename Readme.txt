Open the folder in the command prompt or terminal (cd foldername) and run the below given commands.

1. To run the chatbot, we have two main files; train.py and query_chatbot.py.
2. Priorly few dependencies need to be installed through command prompt or terminal - 
	pip install tensorflow
	pip install keras
	pip install numpy
	pip install nltk
3. Firstly, we train the model using the following command in the terminal or command prompt:
	python train.py
4. If we don’t see any error during training, we have successfully created the model. Then to run the app, run the query_chatbot.py file in the terminal or command prompt.
	python query_chatbot.py
The program will open up a GUI window within a few seconds. With the GUI you can easily chat with the bot or ask any queries to the bot.