##############################################
# Author : Rutvik Trivedi                    #
# Email: rutviktrivedi04@gmail.com           #
# Website: https://Rutvik-Trivedi.github.io  #
# Date: 15 May, 2020                         #
##############################################

from __future__ import division


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import json
import string
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

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
            temp = re.sub(r'&[a-zA-Z0-9]+;?', "", temp)               # Removes words like &nbsp; , &amp; , etc
            temp = re.sub('RT', "", temp)               # Removes 'RT'
            temp = re.sub("'ld", ' would', temp)      # Preprocess chat language
            temp = re.sub("'d", " had", temp)
            temp = re.sub("'ve", " have", temp)
            temp = re.sub("'m", " am", temp)
            temp = re.sub("n't", " not", temp)
            temp = re.sub("won't", "would not", temp)
            temp = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', "", temp)    # Replaces all urls
            if remove_mentions:
                temp = re.sub("@[A-Za-z0-9_]+","",temp)     # Remove all mentions
            if remove_hashtags:
                temp = re.sub('#[A-Za-z0-9_]+', "", temp)   # Remove all hashtags
            temp = puncts.sub("", temp)                     # Remove punctuations
            temp = temp.encode('ascii', 'ignore').decode('ascii')       # Remove emojis and other junk
            temp = re.sub('[0-9]+[a-zA-Z]+', '<UNIT>', temp)        # Replaces words like 15ft, 12cm, 5k with <UNIT>
            temp = re.sub('[0-9]+', '<NUMBER>', temp)               # Replaces all numbers with '<NUMBER>'
            temp = temp.lower()                                     # Converts the data to lowercase
            temp = re.sub(' u ', ' you ', temp)
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

    def clean(self):
        print('Cleaning the data')
        for username in tqdm(self.users):
            try:
                self.clean_data(username)
            except:
                pass
        print('Done cleaning the data')


class Analytics():

    def __init__(self, folder='tweets'):
        self.folder=folder
        # Make a folder to store per user insights
        if not os.path.exists(folder+'/users/'):
            os.mkdir(folder+'/users/')
        if not os.path.exists(folder+'/analytics/'):
            os.mkdir(folder+'/analytics/')

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

    def analyse(self, save=True):
        print('Calculating the analysis of cleaned data')
        l = os.listdir(self.folder+'/cleaned_data/')
        max_words = []
        averages = []
        for username in tqdm(l):
            try:
                analytics = {}
                analytics['name'] = username.strip('.csv')
                analytics['keywords'] = self.extract_keywords(username)
                maximum, average = self.analyse_user(username)
                analytics['max_number_of_words'] = max(maximum)
                analytics['average'] = average
                with open(self.folder+'/analytics/'+username.strip('.csv')+'.json', 'w') as f:
                    json.dump(analytics)
                max_words.append(analytics['max_number_of_words'])
                averages.append(average)
            except:
                pass

        print('Done calculating the analytics')
        ax = plt.subplot(11)
        ax.bar(max_words, list(range(1,len(max_words)+1)), color='b', align='center', label='Maximum Words per User')
        ax.bar(averages, list(range(1,len(averages)+1)), color='g', align='center', label='Average Words per User')
        ax.autoscale(tight=True)
        ax.xlabel('User Count')
        ax.legend(loc='upper left')
        if save:
            print("Saving...")
            plt.savefig('analytics.png')
        else:
            plt.show()
        print("Done")


    def extract_keywords(self, username, max_features=5):
        '''Extracts keywords from the tweets and determines the most tweeted topic by the particular user'''
        corpus = pd.read_csv(self.folder+'/cleaned_data/'+username)['tweets']
        corpus.dropna(inplace=True)
        corpus = corpus.tolist()
        vectorizer = TfidfVectorizer(max_features=max_features)
        _ = vectorizer.fit_transform(corpus)
        return vectorizer.get_feature_names()
