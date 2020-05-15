##############################################
# Author : Rutvik Trivedi                    #
# Email: rutviktrivedi04@gmail.com           #
# Website: https://Rutvik-Trivedi.github.io  #
# Date: 15 May, 2020                         #
##############################################

from __future__ import division


import pandas as pd
import re
import string
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

stop_words = list(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
puncts = re.compile('[%s]' % re.escape(string.punctuation))

class Cleaner():

    def __init__(self, folder='tweets'):
        self.folder=folder
        # Make the list of all available usernames
        self.users = os.listdir(folder)
        # Make folder for storing cleaned data
        if not os.path.exists(folder+'/cleaned_data/'):
            os.mkdir(folder+'/cleaned_data/')


    def clean_data(self, username, remove_hashtags=True, remove_mentions=True, remove_stopwords=True):
        # Read the CSV File. Helps if the CSV file is messed up
        with open(self.folder+'/'+username, 'r') as f:
            data = f.readlines()[1:]
        for i in range(len(data)):
            data[i] = data[i].strip('\n')
            data[i] = ','.join(data[i].split(',')[2:])

        # Preprocessing start
        for i in range(len(data)):
            temp = data[i]
            temp = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', "", temp)    # Replaces all urls
            if remove_mentions:
                temp = re.sub("@[A-Za-z0-9_]+","",temp)     # Remove all mentions
            if remove_hashtags:
                temp = re.sub('#[A-Za-z0-9_]+', "", temp)   # Remove all hashtags
            temp = puncts.sub("", temp)                     # Remove punctuations
            temp = temp.encode('ascii', 'ignore').decode('ascii')       # Remove emojis and other junk
            data[i] = temp
            cleaned_list = []
            tokens = word_tokenize(data[i])                 # Tokenize data
            if remove_stopwords:                # Removes Stopwords
                for j in tokens:
                    if j not in stop_words:
                        cleaned_list.append(lemmatizer.lemmatize(j))                # Lemmatize. Can also try stemming
            else:                               # Doesn't remove stopwords
                for j in tokens:
                    cleaned_list.append(lemmatizer.lemmatize(j))                    # Lemmatize. Can also try Stemming
            data[i] = ' '.join(cleaned_list)

        cleaned_df= pd.DataFrame({'tweets':data})    #Create a personal CSV for all users
        cleaned_df.to_csv(self.folder+'/cleaned_data/'+username, index=False)


class Analytics():

    def __init__(self, folder='tweets'):
        self.folder=folder
        # Make a folder to store per user insights
        if not os.path.exists(folder+'/users/'):
            os.mkdir(folder+'/users/')

    def analyse_user(self, username):
        # Analyse for each user
        data = pd.read_csv(self.folder+'/cleaned_data/'+username)['tweets'].tolist()
        max_word_counts = []
        for tweet in data:
            try:
                max_word_counts.append(len(word_tokenize(tweet)))
            except TypeError:           # Occurs if string is empty
                max_word_counts.append(0)
        max_word_counts = sorted(max_word_counts)
        average = sum(max_word_counts)/len(max_word_counts)
        return max_word_counts, average

    def plot_data(self, username, save=True):
        # Plot data
        max_word_counts, average = self.analyse_user(username)
        plt.bar(list(range(1,len(max_word_counts)+1)), max_word_counts)
        plt.xlabel("Tweet Number")
        plt.ylabel("Maximum number of words")
        plt.title("Maximum words per tweet for {}".format(username.strip('.csv')))
        if save:
            plt.savefig(self.folder+'/users/'+username.strip('.csv')+'.png')
        else:
            plt.show()
