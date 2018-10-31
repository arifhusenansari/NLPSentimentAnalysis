# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:10:57 2018

@author: user
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import numpy as np
import pandas as pd
import os
import re
import pickle


review_train = []
review_test = []



#-- This function will iterate through folder.
#-- Read data from train and test folder and also tag review as positive or negetive.

def read_data_from_directory (main_dir_path):
    review_train = []
    review_test = []
    new_review=[]
    dir_list = os.listdir(main_dir_path)
    for directory in dir_list:
        if directory == 'train':
            sub_dir_list= os.listdir(main_dir_path+'\\'+directory)
            #-- If sub directory is 'pos'. 
            #-- It will read file.
            for sub_dir in sub_dir_list:
                if sub_dir == 'pos':
                    files = os.listdir(main_dir_path+'\\'+directory+'\\'+sub_dir)
                    for file in files:
                        for line in open(main_dir_path+'\\'+directory+'\\'+sub_dir+'\\'+file,'r',encoding="utf8"):
                            review_train.append((line,'pos'))                                
                elif sub_dir == 'neg':
                    files = os.listdir(main_dir_path+'\\'+directory+'\\'+sub_dir)
                    for file in files:
                        for line in open(main_dir_path+'\\'+directory+'\\'+sub_dir+'\\'+file,'r',encoding="utf8"):
                                review_train.append((line,'neg'))                                
                        
        elif directory == 'test':
            sub_dir_list= os.listdir(main_dir_path+'\\'+directory)
            #-- If sub directory is 'pos'. 
            #-- It will read file.
            for sub_dir in sub_dir_list:
                if sub_dir == 'pos':
                    files = os.listdir(main_dir_path+'\\'+directory+'\\'+sub_dir)
                    for file in files:
                        for line in open(main_dir_path+'\\'+directory+'\\'+sub_dir+'\\'+file,'r',encoding="utf8"):
                            review_test.append((line,'pos'))                                
                elif sub_dir == 'neg':
                    files = os.listdir(main_dir_path+'\\'+directory+'\\'+sub_dir)
                    for file in files:
                        for line in open(main_dir_path+'\\'+directory+'\\'+sub_dir+'\\'+file,'r',encoding="utf8"):
                                review_test.append((line,'neg')) 
        elif directory == 'new':
            sub_dir_list= os.listdir(main_dir_path+'\\'+directory)
            #-- It will read file.
            for file in sub_dir_list:
                for line in open(main_dir_path+'\\'+directory+'\\'+file,'r',encoding="utf8"):
                    new_review.append((line,''))                                
                                       
    return review_train,review_test,new_review

def process_text_data(data):
    REPLACE_WITH_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    
    data_clean= []
    target = []
    for tup in data:
        review = tup[0]
        tag= tup[1]
        review = REPLACE_WITH_NO_SPACE.sub("",review.lower())
        review = REPLACE_WITH_SPACE.sub(" ",review)
        data_clean.append(review)
        target.append(tag)
    return data_clean,target

def vectories_data(review_train_clean,review_test_clean):
    #-- vectorization is used to make review compatible for machine learning algo.
    #-- it will generate column for each and every word. 
    #-- Put value 1 if world avaialable else 0.
    #-- While generating vectorization form of data. We only use train data to consider words.
    #-- Hence we fit data on train and use that model to build vectorization form for test.
    cv = CountVectorizer(binary = True)
    cv.fit(review_train_clean)
    x = cv.transform(review_train_clean)
    x_test = cv.transform(review_test_clean)
    return x,x_test,cv;

def train_classifier (x,train_target):
    #-- Spit data in test and train.
    x_train,x_val,y_train,y_val = train_test_split(x,train_target,train_size= 0.75)            
    
    #-- Fit model for different values of C
    for c in [0.01,0.05,0.25,0.5,1]:
        lr = LogisticRegression(C=c)
        lr.fit(x_train,y_train)
        accuracy = accuracy_score(y_val,lr.predict(x_val))
        print ("Accuracy for C= %s: %s"%(c,accuracy))
    
def final_train_classifier (x,train_target):
    lr = LogisticRegression(C=0.5)
    lr.fit(x,train_target)
    return lr
def save_model ( path, model,cv,filename):
    filepath= path+"\\"+filename
    pickle.dump(model, open (filepath+'.sav', 'wb'))
    pickle.dump(cv,open(filepath+'_cv.sav','wb'))
    

def load_model(path,filename):
    filepath= path+"\\"+filename
    return pickle.load(open(filepath+'.sav','rb')),pickle.load(open(filepath+'_cv.sav','rb'))

def print_best_discriminating_word():
    #-- Find out top 10 discriminating word for positive and negetive sentiment.
    #-- This will be based on the coefficient for different word.
    
    #-- Get the list of features from CountVectorizer.
    #print(cv.get_feature_names())
    
    feature_to_coef = {
                word:coef for word, coef in zip(cv.get_feature_names(),finalmodel.coef_[0])
            }
    #-- Print top 10 discriminationg positive word.
    #-- Sorted dictionary based on coefficient.
    
    for best_positive in sorted(feature_to_coef.items(),key=lambda x: x[1],reverse=True)[:5]:
        print (best_positive)
        
    for best_positive in sorted(feature_to_coef.items(),key=lambda x: x[1],reverse=False)[:5]:
        print (best_positive)
        
if __name__ == "__main__":
    
    is_train = input('Do you want to train on new data? Press [y|Y] or [n|N]\n')
    main_dir_path = "E:\\Arif\\Work\\Data Science Course\\DataScienceProjects\\SentimentAnalysisNLP\\aclImdb"    
    model_path = "E:\\Arif\\Work\\Data Science Course\\DataScienceProjects\\SentimentAnalysisNLP\\"
    if is_train == 'Y' or is_train == 'y':        
        #-- Read data from main direcotry.
        review_train,review_test,_ = read_data_from_directory(main_dir_path=main_dir_path)
        
        #-- Process text data to remove unused information and covert to lower case.
        review_train_clean,train_target = process_text_data(review_train)
        review_test_clean,test_target = process_text_data(review_train)
    
        
        #-- Cross check cleaning process
    #    print(review_train_clean[0])
    #    print(train_target[0])
        
        #-- Convert target to 0 and 1 format
        train_target = [int(a == 'pos') for a in train_target]
        test_target  = [int(a == 'pos') for a in test_target ]
        
        #-- Vectorise the data. It will check whether word in available in the review or not.
        x,x_test,cv = vectories_data(review_train_clean,review_test_clean)
        
        #-- train model and find accuracy for different value of c
        train_classifier(x,train_target)
        
        #-- As per model efficency. 
        #-- for c= 0.05 accuracy is: 0.88
        
        #-- Finally train model using c= 0.05
        finalmodel = final_train_classifier(x,train_target)
        #-- Final Model efficiency on Whole train data set.
        print('Accuracy on final model for Train data is %s:'%(accuracy_score(train_target,finalmodel.predict(x))))
        #-- Final Model efficiency on Whole Test data set.
        print('Accuracy on final model for test data is %s:'%(accuracy_score(test_target,finalmodel.predict(x_test))))
        #-- Save model for future use.
        save_model(path=model_path , model=finalmodel,cv=cv,filename='finalized_model')
        
    elif is_train == 'n' or is_train == 'N':        
        #-- Load model and predict data.
        stored_model,cv = load_model(path=model_path,filename='finalized_model')
        _,_,new_data = read_data_from_directory(main_dir_path=main_dir_path,)
        new_data_clean,_= process_text_data(new_data )
        new_data_clean_vector = cv.transform(new_data_clean)
        prediction = stored_model.predict(new_data_clean_vector)
        for pred in (prediction):
            if pred == 0:
                print('Negetive')
            elif pred == 1:
                print('Positive')            
        
#        print('Accuracy on final model for test data is %s:'%(stored_model.score(x_test,test_target)))
        
    

    
    
    
    
    
    
        
        
    
    