# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:29:51 2020

@author: Siri
"""

import pandas as pd
import numpy as np
import os 

os.chdir('E:/analytics/projects/self/customer_review_sentiment_analysis')

os.listdir()

restaurant_reviews = pd.read_csv('Restaurant_reviews.tsv',delimiter='\t')

restaurant_reviews.head(5)
restaurant_reviews.columns

# data preperation

import nltk
import re
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
ps = PorterStemmer()
review_final = list()
for i in range(len(restaurant_reviews['Review'])) :
    # removing special characters
    review = re.sub(pattern='[^a-zA-Z]',repl=' ',string=restaurant_reviews['Review'][i])
    # lowering the sentences
    review = review.lower()
    # slitting review into words
    review_words = review.split()
    # Removing stopwords 
    review_words = [rev_word for rev_word in review_words if rev_word not in stopwords.words('english')]
    # Stemming
    stem_words = [ps.stem(word) for word in review_words]
    #forming sentences again
    review_stem = ' '.join(stem_words)
    # adding the review to final list 
    review_final.append(review_stem)
    
review_final[0:5]    


# Bag of words and feature target splitting

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
file_name ='transform.pkl'
pickle.dump(cv,open(file_name,'wb'))
x = cv.fit_transform(review_final)
y = restaurant_reviews['Liked']
y.shape

# splitting into train and test

from sklearn.model_selection import train_test_split

x_train, x_test , y_train , y_test= train_test_split(x,y,test_size=0.2,random_state=40)

# training using naive bayes theorem

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(x_train,y_train)

y_pred = nb_model.predict(x_test)

# evaluation metrics

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

print(confusion_matrix(y_test,y_pred))
# [[82 24]
#  [14 81]]

print(accuracy_score(y_test,y_pred)) #0.8109452736318408

print(precision_score(y_test,y_pred)) #0.7714285714285715

print(f1_score(y_test,y_pred)) #0.81

print(recall_score(y_test,y_pred))  #0.8526315789473684

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf =TfidfVectorizer()
x_tf_idf = tf_idf.fit_transform(review_final).toarray()
x_train_tf,x_test_tf,y_train_tf,y_test_tf = train_test_split(x_tf_idf,y,test_size=0.2,random_state=40)

mod_tf_idf = MultinomialNB()
mod_tf_idf.fit(x_train_tf,y_train_tf)
mod_tf_idf.score(x_train_tf,y_train_tf)

y_pred_tf = mod_tf_idf.predict(x_test_tf)

confusion_matrix(y_test_tf,y_pred_tf) 
# [[80, 26],
#        [14, 81]]
accuracy_score(y_test_tf,y_pred_tf)  #  0.8009950248756219

# hyper parameter tuning for bag of words 

alpha = np.arange(0.1,1.1,0.1)
parameters = {'alpha':alpha}
tune_mod = MultinomialNB()
from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(estimator= tune_mod , param_grid=parameters,cv=10)
grid_model.fit(x_train,y_train)
grid_model.best_params_  #0.3
grid_model.best_score_   #0.77

y_pred_tune = grid_model.predict(x_test)
confusion_matrix(y_test,y_pred_tune)
# [[85, 21],
# [17, 78]]
accuracy_score(y_test,y_pred_tune) #0.8109452

tuned_model = MultinomialNB(alpha=0.3)
tuned_model.fit(x_train,y_train)
y_pred_tuned = tuned_model.predict(x_test)
confusion_matrix(y_test,y_pred_tuned)

import pickle

file_name = 'customer_review_sentiment_analysis.pkl'
pickle.dump(tuned_model,open(file_name,'wb'))

def predict_sentiment(review) :
    review = re.sub(pattern='[^a-zA-Z]',repl=' ',string=review)
    # lowering the sentences
    review = review.lower()
    # slitting review into words
    review_words = review.split()
    # Removing stopwords 
    review_words = [rev_word for rev_word in review_words if rev_word not in stopwords.words('english')]
    # Stemming
    stem_words = [ps.stem(word) for word in review_words]
    #forming sentences again
    review_stem = ' '.join(stem_words)
    # adding the review to final list 
    review_sentiment = cv.transform([review_stem])
    return tuned_model.predict(review_sentiment)
    
if predict_sentiment('the food is good') : 
    print('The review is good review')
else :
    print('The review is bad review')


if predict_sentiment('the food color is good') : 
    print('The review is good review')
else :
    print('The review is bad review')

    def predict_sentiment(review) :
                ps=PorterStemmer()
                review = re.sub(pattern='[^a-zA-Z]',repl=' ',string=review)
                # lowering the sentences
                review = review.lower()
                # slitting review into words
                review_words = review.split()
                # Removing stopwords 
                review_words = [rev_word for rev_word in review_words if rev_word not in stopwords.words('english')]
                # Stemming
                stem_words = [ps.stem(word) for word in review_words]
                #forming sentences again
                review_stem = ' '.join(stem_words)
                review_sentiment = cv.transform([review_stem])
                return model.predict(review_sentiment)
        if predict_sentiment('the food is good') : 
            return render_template('index.html',prediction_text='The review is good review')
        else :
            return render_template('index.html',prediction_text='The review is bad review')
        