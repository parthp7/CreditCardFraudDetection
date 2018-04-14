#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:10:59 2018

@author: parth
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import time


def normalize(X):
    
    for feature in X.columns:
        X[feature] -= X[feature].mean()
        X[feature] /= X[feature].std()
        
    return X


def undetectedFraudRate(test, pred):
    
    det = 0
    fraud = 0
    
    for i in range(0, len(test)):
        
        if(test[i]==1):
            fraud = fraud + 1
            if(pred[i]==1):
                det = det + 1
                
    return det, fraud


if __name__ == "__main__":
    
    st_time = time.time()
    data = pd.read_csv('creditcard.csv')
    
    features = ['Amount'] + ['V%d' % number for number in range (1,29)]
    target = 'Class'
    
    X = data[features]
    Y = data[target]
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    
    model = GaussianNB()
    
    for train_indices, test_indices in splitter.split(X,Y):
        
        X_train, Y_train = X.iloc[train_indices], Y.iloc[train_indices]
        X_test, Y_test = X.iloc[test_indices], Y.iloc[test_indices]
        
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        
        print(classification_report(Y_test, predicted))
        
        det, tot = undetectedFraudRate(Y_test.values, predicted)
        print('Fraud detection accuracy: {}'.format(det/tot*100))
        
    print('\nExecution time is: {} seconds'.format(time.time()-st_time))