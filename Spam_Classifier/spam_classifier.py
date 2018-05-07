
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:20:07 2018
@author: ritu
"""

#Importing Libraries
import pandas as pd

#-----Importing the dataset using read_table in pandas---------

#This returns a dataframe: a 2D labelled data structure with columns of 
#potentially different types
dataset = pd.read_table(filepath_or_buffer = 'SMSSpamCollection',sep = '\t', 
              names = ['label','sms_message'])

#dataset.head(n) returns the n rows of the dataframe
print(dataset.head())

#------Data Preprocessing------------

#Converting labels to numerical values

test = dataset['label']

def func(x):
    if (x == 'ham'):
        return 0
    elif(x == 'spam'):
        return 1
        
dataset['label'] = dataset.label.map(func)

#Bag of Words using scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
bag_of_words = CountVectorizer(input = "dataset['sms_message']",
                               lowercase = True, stop_words='english', 
                               token_pattern=r"(?u)\b\w\w+\b")

#print(bag_of_words)

bag_1 = bag_of_words.fit(dataset['sms_message'])
features = bag_of_words.get_feature_names()

bag_2 = bag_of_words.transform(dataset['sms_message'])

print(bag_2.toarray())

frequency_matrix =pd.DataFrame(data = bag_2.toarray(),columns = features)

#Training and Testing Sets

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset['sms_message'], 
                                                    dataset['label'], 
                                                    random_state=1)
"""
print('Number of rows in the total set: {}'.format(dataset.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
"""

# Applying Bag of Words to the splitted dataset
bag_of_words = CountVectorizer()

#print(bag_of_words)

training_data = bag_of_words.fit_transform(X_train)
testing_data = bag_of_words.transform(X_test)

#Implementing Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)

predictions = naive_bayes.predict(testing_data)


#Accuracy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('precision_score score: ', format(precision_score(y_test,predictions)))
print('recall_score score: ', format(recall_score(y_test,predictions)))
print('f1_score score: ', format(f1_score(y_test,predictions)))






















