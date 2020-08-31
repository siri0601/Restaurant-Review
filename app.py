# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:25:47 2020

@author: HP
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)
model = pickle.load(open('customer_review_sentiment_analysis.pkl', 'rb'))
cv = pickle.load(open('transform.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
    
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        review_input = request.form['review']        
        def predict_sentiment(review_stmt) :
            ps=PorterStemmer()
            review = re.sub(pattern='[^a-zA-Z]',repl=' ',string=review_stmt)
            # lowering the sentences
            review = review.lower()
            # slitting review into words
            review_words = review.split()
            # Removing stopwords 
            review_words = [rev_word for rev_word in review_words if rev_word not in stopwords.words('english')]
            if len(review_words) == 0 :
                return render_template('index.html',prediction_text='Please enter the review')
            # Stemming
            stem_words = [ps.stem(word) for word in review_words]
            #forming sentences again
            review_stem = ' '.join(stem_words)
            review_sentiment = cv.transform([review_stem])
            return model.predict(review_sentiment)
        sentiment = predict_sentiment(review_input)
        if sentiment == 1 : 
            return render_template('index.html',prediction_text='The review is good review')
        elif sentiment== 0 :
            return render_template('index.html',prediction_text='The review is bad review')
        else:
            return render_template('index.html',prediction_text='Please enter the review')
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)