#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:20:14 2018

@author: skondaveeti
"""

import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import math
import pickle


class Model:
    
    def __init__(self):
        pass
        
    """
    Parameter1: csv_file = Pass the absolute path of the csv file to train the efficient model. Ex:shoe_purchases.csv
    
    Example to call the function :  Model_Score_obj = Model()
                                    Model_Score_obj.train('shoe_purchases.csv')
    """
    def train(self, csv_file):
        #dataset = pd.read_csv('shoe_purchases.csv')
        dataset = pd.read_csv(csv_file)


        dataset['shoe_style'].replace('a', 0,inplace=True)
        dataset['shoe_style'].replace('m', 1,inplace=True)
        
        dataset = dataset.rename(index=str, columns={"Unnamed: 0": "Shoe_Purchases"})
        
        
        X = dataset.iloc[:, 1:4].values
        y = dataset.iloc[:, 4:7].values
        
        
        ###########Independent Variable (Numerical)###########
        print ('EDA of the data passed.')
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(dataset['length']);
        
        plt.figure(2)
        plt.subplot(131)
        sns.distplot(dataset['width']);
        
        plt.subplot(132)
        sns.distplot(dataset['arch_height']);
        
        
        plt.show()
        
        ############ Correlation ############
        
        matrix = dataset.corr()
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
        
        
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        
        ########## Missing Value Treatment ################
        
        dataset['Shoe_Purchases'].fillna(dataset['Shoe_Purchases'].mode()[0], inplace=True)
        dataset['length'].fillna(dataset['length'].mode()[0], inplace=True)
        dataset['width'].fillna(dataset['width'].mode()[0], inplace=True)
        dataset['arch_height'].fillna(dataset['arch_height'].mode()[0], inplace=True)
        
        
        
        
        # Fitting RandomForestRegressor to the dataset
        max_depth = 30
        regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                                  max_depth=max_depth,
                                                                  random_state=0))
        regr_multirf.fit(X_train, y_train)
        
        save_regressor = open("regressor_model.pickle","wb")
        pickle.dump(regr_multirf, save_regressor)
        save_regressor.close()
    
    """
    Parameter1: csv_file = Pass the absolute path of the csv file to predict the result for. Ex:shoe_purchases.csv
    
    Example to call the function :  Model_Score_obj = Model()
                                    Model_Score_obj.predict('shoe_purchases.csv')

    Outputs the Predicted values to submit_file.csv
    """
    def predict(self, csv_file):
        #predict_dataset = pd.read_csv('shoe_purchases.csv')
        predict_dataset = pd.read_csv(csv_file)
        
        regressor_f = open("regressor_model.pickle", "rb")
        pickled_regressor = pickle.load(regressor_f)
        regressor_f.close()
        
        # Predict on new data
        y_multirfed = pickled_regressor.predict(predict_dataset)
        for i, s in enumerate(y_multirfed[:,0]):
            y_multirfed[:,0][i] = round(y_multirfed[:,0][i], 1)
            if y_multirfed[:,0][i] > 0.5:
                y_multirfed[:,0][i] = 1
            else:
                y_multirfed[:,0][i] = 0
        
        
        for i, s in enumerate(y_multirfed[:,1]):
            y_multirfed[:,1][i] = round(y_multirfed[:,1][i], 1)
            ab = list(math.modf(y_multirfed[:,1][i]))
            if ab[0] > 0.5 and ab[0] < 0.7:
                ab[0] = 0.5
            elif ab[0] > 0.3 and ab[0] < 0.5:
                ab[0] = 0.5
            elif ab[0] > 0.7:
                ab[0] = 1
            else:
                ab[0] = 0
            y_multirfed[:,1][i] = ab[0]+ab[1]     
        
        
        for i, s in enumerate(y_multirfed[:,2]):
            y_multirfed[:,2][i] = int(y_multirfed[:,2][i])
        
        df = pd.DataFrame({'shoe_style':y_multirfed[:,0], 'length_size': y_multirfed[:,1], 'width_size': y_multirfed[:,2]})
        df.to_csv("submit_file.csv", index = False, encoding='utf-8')
        return "Executed Succesfullly!!"



if __name__== "__main__":
    Model_Score_obj = Model()
    Model_Score_obj.train('shoe_purchases.csv')
    Model_Score_obj.predict('Test.csv')

