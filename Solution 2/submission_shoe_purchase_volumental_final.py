#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:11:44 2018

@author: skondaveeti
"""

import pandas as pd
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import pickle
from sklearn.linear_model import LinearRegression
import xgboost
from math import sqrt


class Model:
    

    def __init__(self):
        pass
        
    """
    Parameter1: csv_file = Pass the absolute path of the csv file to train the efficient model. Ex:shoe_purchases.csv
    Parameter2: retrain # To retrain by fitting model again.
    
    Example to call the function :  Model_Score_obj = Model()
                                    Model_Score_obj.train('shoe_purchases.csv')
                                    
                                    (or)
                                    
                                    Model_Score_obj = Model()
                                    Model_Score_obj.train('shoe_purchases.csv', retrain = False/True)
    """
    # Arg csv_file = Pass the absolute path of the csv file.
    def train(self, csv_file, retrain = False):
        dataset = pd.read_csv(csv_file)
        dataset['shoe_style'].replace('a', 0,inplace=True)
        dataset['shoe_style'].replace('m', 1,inplace=True)
        
        dataset = dataset.rename(index=str, columns={"Unnamed: 0": "Shoe_Purchases"})
        
        
        X = dataset.iloc[:, 1:4].values
        y_shoe_style= dataset.iloc[:, 4].values
        y_length_size= dataset.iloc[:, 5].values
        y_width_size= dataset.iloc[:, 6].values
        
        
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
        X_train_shoe_style, X_test_shoe_style, y_train_shoe_style, y_test_shoe_style = train_test_split(X, y_shoe_style, test_size = 0.2, random_state = 0)
        X_train_length_size, X_test_length_size, y_train_length_size, y_test_length_size = train_test_split(X, y_length_size, test_size = 0.2, random_state = 0)
        X_train_width_size, X_test_width_size, y_train_width_size, y_test_width_size = train_test_split(X, y_width_size, test_size = 0.2, random_state = 0)
        
        
        
        ########## Missing Value Treatment ################
        
        dataset['Shoe_Purchases'].fillna(dataset['Shoe_Purchases'].mode()[0], inplace=True)
        dataset['length'].fillna(dataset['length'].mode()[0], inplace=True)
        dataset['width'].fillna(dataset['width'].mode()[0], inplace=True)
        dataset['arch_height'].fillna(dataset['arch_height'].mode()[0], inplace=True)
        
        # Initializing the LinearRegression models
        linear_regressor_shoe_style = LinearRegression()
        linear_regressor_length_size = LinearRegression()
        linear_regressor_width_size = LinearRegression()
        
        # Initializing the RandomForestRegressor models
        random_forest_regressor_shoe_style = RandomForestRegressor(n_estimators = 10, random_state = 0)
        random_forest_regressor_length_size = RandomForestRegressor(n_estimators = 10, random_state = 0)
        random_forest_regressor_width_size = RandomForestRegressor(n_estimators = 10, random_state = 0)
        
        # Initializing the xgboost models
        xgb_regressor_shoe_style = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
        
        xgb_regressor_length_size = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
        xgb_regressor_width_size = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
        
        
        # Fitting Models to the dataset
        if retrain == False:
            i = 1
        else:
            i = 2
        for i in range(0,i):
            print('-----------------------------------------------------------------------')
            print('Running the model:',i)
            # Fitting model with multiple linear Regressor for shoe_style
            
            linear_regressor_shoe_style.fit(X_train_shoe_style, y_train_shoe_style)
            # Fitting model with multiple linear Regressor for length_size
            
            linear_regressor_length_size.fit(X_train_length_size, y_train_length_size)
            # Fitting model with multiple linear Regressor for width_size
            
            linear_regressor_width_size.fit(X_train_width_size, y_train_width_size)
            
            # Fitting model with Random Forest Regressor for shoe_style
            # random_forest_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
            
            random_forest_regressor_shoe_style.fit(X_train_shoe_style, y_train_shoe_style)
            
            
             # Fitting model with Random Forest Regressor for length_size
            
            random_forest_regressor_length_size.fit(X_train_length_size, y_train_length_size)
            
            
             # Fitting model with Random Forest Regressor for width_size
            
            random_forest_regressor_width_size.fit(X_train_width_size, y_train_width_size)
            
            
            
            # Fitting model with XGBOOST Regressor for shoe_style
            
            xgb_regressor_shoe_style.fit(X_train_shoe_style, y_train_shoe_style)
            
            # Fitting model with XGBOOST Regressor for length_size
            
            xgb_regressor_length_size.fit(X_train_length_size, y_train_length_size)
            
            # Fitting model with XGBOOST Regressor for width_size
            
            xgb_regressor_width_size.fit(X_train_width_size, y_train_width_size)
        
        
        # Predict on new data with linear regressors
        y_linear_regressor_shoe_style = linear_regressor_shoe_style.predict(X_test_shoe_style)
        y_linear_regressor_length_size = linear_regressor_length_size.predict(X_test_length_size)
        y_linear_regressor_width_size = linear_regressor_width_size.predict(X_test_width_size)
        
        # Predict on new data with linear regressors
        y_random_forest_regressor_shoe_style = random_forest_regressor_shoe_style.predict(X_test_shoe_style)
        y_random_forest_regressor_length_size = random_forest_regressor_length_size.predict(X_test_length_size)
        y_random_forest_regressor_width_size = random_forest_regressor_width_size.predict(X_test_width_size)
        
        # Predict on new data with linear regressors
        y_xgb_regressor_shoe_style = xgb_regressor_shoe_style.predict(X_test_shoe_style)
        y_xgb_regressor_length_size = xgb_regressor_length_size.predict(X_test_length_size)
        y_xgb_regressor_width_size = xgb_regressor_width_size.predict(X_test_width_size)
        
        
        #Formatting for Linear Regressor
        for i, s in enumerate(y_linear_regressor_shoe_style):
            y_linear_regressor_shoe_style[i] = round(y_linear_regressor_shoe_style[i], 1)
            if y_linear_regressor_shoe_style[i] > 0.5:
                y_linear_regressor_shoe_style[i] = 1
            else:
                y_linear_regressor_shoe_style[i] = 0
        
        
        for i, s in enumerate(y_linear_regressor_length_size):
            y_linear_regressor_length_size[i] = round(y_linear_regressor_length_size[i], 1)
            ab = list(math.modf(y_linear_regressor_length_size[i]))
            if ab[0] > 0.5 and ab[0] < 0.7:
                ab[0] = 0.5
            elif ab[0] > 0.3 and ab[0] < 0.5:
                ab[0] = 0.5
            elif ab[0] > 0.7:
                ab[0] = 1
            else:
                ab[0] = 0
            y_linear_regressor_length_size[i] = ab[0]+ab[1]     
        
        
        for i, s in enumerate(y_linear_regressor_width_size):
            y_linear_regressor_width_size[i] = int(y_linear_regressor_width_size[i])
        
        #Formatting for Random Forest
        for i, s in enumerate(y_random_forest_regressor_shoe_style):
            y_random_forest_regressor_shoe_style[i] = round(y_random_forest_regressor_shoe_style[i], 1)
            if y_random_forest_regressor_shoe_style[i] > 0.5:
                y_random_forest_regressor_shoe_style[i] = 1
            else:
                y_random_forest_regressor_shoe_style[i] = 0
        
        
        for i, s in enumerate(y_random_forest_regressor_length_size):
            y_random_forest_regressor_length_size[i] = round(y_random_forest_regressor_length_size[i], 1)
            ab = list(math.modf(y_random_forest_regressor_length_size[i]))
            if ab[0] > 0.5 and ab[0] < 0.7:
                ab[0] = 0.5
            elif ab[0] > 0.3 and ab[0] < 0.5:
                ab[0] = 0.5
            elif ab[0] > 0.7:
                ab[0] = 1
            else:
                ab[0] = 0
            y_random_forest_regressor_length_size[i] = ab[0]+ab[1]     
        
        
        for i, s in enumerate(y_random_forest_regressor_width_size):
            y_random_forest_regressor_width_size[i] = int(y_random_forest_regressor_width_size[i])
            
        #Formatting for XGBoost 
        for i, s in enumerate(y_xgb_regressor_shoe_style):
            y_xgb_regressor_shoe_style[i] = round(y_xgb_regressor_shoe_style[i], 1)
            if y_xgb_regressor_shoe_style[i] > 0.5:
                y_xgb_regressor_shoe_style[i] = 1
            else:
                y_xgb_regressor_shoe_style[i] = 0
        
        
        for i, s in enumerate(y_xgb_regressor_length_size):
            y_xgb_regressor_length_size[i] = round(y_xgb_regressor_length_size[i], 1)
            ab = list(math.modf(y_xgb_regressor_length_size[i]))
            if ab[0] > 0.5 and ab[0] < 0.7:
                ab[0] = 0.5
            elif ab[0] > 0.3 and ab[0] < 0.5:
                ab[0] = 0.5
            elif ab[0] > 0.7:
                ab[0] = 1
            else:
                ab[0] = 0
            y_xgb_regressor_length_size[i] = ab[0]+ab[1]     
        
        
        for i, s in enumerate(y_xgb_regressor_width_size):
            y_xgb_regressor_width_size[i] = int(y_xgb_regressor_width_size[i])

        
        # Calculating the RMSE (Root Mean Squared Error)
        print('-----------------------------------------------------------------------')
        print('Root Mean Squared Error')
        rms_linear_regressor_shoe_style = sqrt(mean_squared_error(y_test_shoe_style, y_linear_regressor_shoe_style))
        print("rms_linear_regressor_shoe_style:  ", rms_linear_regressor_shoe_style)
        rms_linear_regressor_length_size = sqrt(mean_squared_error(y_test_length_size, y_linear_regressor_length_size))
        print("rms_linear_regressor_length_size:  ", rms_linear_regressor_length_size)
        rms_linear_regressor_width_size = sqrt(mean_squared_error(y_test_width_size, y_linear_regressor_width_size))
        print("rms_linear_regressor_width_size:  ", rms_linear_regressor_width_size)
        
        print('-----------------------------------------------------------------------')
        rms_random_forest_regressor_shoe_style = sqrt(mean_squared_error(y_test_shoe_style, y_random_forest_regressor_shoe_style))
        print("rms_random_forest_regressor_shoe_style:  ", rms_random_forest_regressor_shoe_style)
        rms_random_forest_regressor_length_size = sqrt(mean_squared_error(y_test_length_size, y_random_forest_regressor_length_size))
        print("rms_random_forest_regressor_length_size:  ", rms_random_forest_regressor_length_size)
        rms_random_forest_regressor_width_size = sqrt(mean_squared_error(y_test_width_size, y_random_forest_regressor_width_size))
        print("rms_random_forest_regressor_width_size:  ", rms_random_forest_regressor_width_size)
              
        print('-----------------------------------------------------------------------')
        rms_xgb_regressor_shoe_style = sqrt(mean_squared_error(y_test_shoe_style, y_xgb_regressor_shoe_style))
        print("rms_xgb_regressor_shoe_style:  ", rms_xgb_regressor_shoe_style)
        rms_xgb_regressor_length_size = sqrt(mean_squared_error(y_test_length_size, y_xgb_regressor_length_size))
        print("rms_xgb_regressor_length_size:  ", rms_xgb_regressor_length_size)
        rms_xgb_regressor_width_size = sqrt(mean_squared_error(y_test_width_size, y_xgb_regressor_width_size))
        print("rms_xgb_regressor_width_size:  ", rms_xgb_regressor_width_size)
        print('-----------------------------------------------------------------------')
        
        rmse_dataframe = [[rms_linear_regressor_shoe_style,rms_random_forest_regressor_shoe_style,rms_xgb_regressor_shoe_style],
                     [rms_linear_regressor_length_size,rms_random_forest_regressor_length_size,rms_xgb_regressor_length_size],
                     [rms_linear_regressor_width_size,rms_random_forest_regressor_width_size,rms_xgb_regressor_width_size]]
        
        # Plotting the RMSE
        plt.figure()
        plt.boxplot(rmse_dataframe, 1)
        
        # Since the RMSE is low for Linear Regression We will use that for further prediction.
        # Pickling the linear regressors and saving it to the filesystem.
        save_linear_regressor_shoe_style = open("linear_regressor_shoe_style.pickle","wb")
        pickle.dump(linear_regressor_shoe_style, save_linear_regressor_shoe_style)
        save_linear_regressor_shoe_style.close()
        
        save_linear_regressor_length_size = open("linear_regressor_length_size.pickle","wb")
        pickle.dump(linear_regressor_length_size, save_linear_regressor_length_size)
        save_linear_regressor_length_size.close()
        
        save_linear_regressor_width_size = open("linear_regressor_width_size.pickle","wb")
        pickle.dump(linear_regressor_width_size, save_linear_regressor_width_size)
        save_linear_regressor_width_size.close()
        
        
    """
    Parameter1: csv_file = Pass the absolute path of the csv file to predict the result for. Ex:shoe_purchases.csv
    
    Example to call the function :  Model_Score_obj = Model()
                                    Model_Score_obj.predict('shoe_purchases.csv')

    Outputs the Predicted values to submit_file.csv
    """
    def predict(self, csv_file):
        #predict_dataset = pd.read_csv('shoe_purchases.csv')
        predict_dataset = pd.read_csv(csv_file)
        
        # Unpickling the linear regressors.
        open_linear_regressor_shoe_style = open("linear_regressor_shoe_style.pickle", "rb")
        pickled_open_linear_regressor_shoe_style = pickle.load(open_linear_regressor_shoe_style)
        open_linear_regressor_shoe_style.close()
        
        open_linear_regressor_length_size = open("linear_regressor_length_size.pickle", "rb")
        pickled_open_linear_regressor_length_size = pickle.load(open_linear_regressor_length_size)
        open_linear_regressor_length_size.close()
        
        open_linear_regressor_width_size = open("linear_regressor_width_size.pickle", "rb")
        pickled_open_linear_regressor_width_size = pickle.load(open_linear_regressor_width_size)
        open_linear_regressor_width_size.close()

        
        # Predict on new data with linear regressors
        predict_y_linear_regressor_shoe_style = pickled_open_linear_regressor_shoe_style.predict(predict_dataset)
        predict_y_linear_regressor_length_size = pickled_open_linear_regressor_length_size.predict(predict_dataset)
        predict_y_linear_regressor_width_size = pickled_open_linear_regressor_width_size.predict(predict_dataset)
        
        #Formatting prediction values for Linear Regression.
        for i, s in enumerate(predict_y_linear_regressor_shoe_style):
            predict_y_linear_regressor_shoe_style[i] = round(predict_y_linear_regressor_shoe_style[i], 1)
            if predict_y_linear_regressor_shoe_style[i] > 0.5:
                predict_y_linear_regressor_shoe_style[i] = 1
            else:
                predict_y_linear_regressor_shoe_style[i] = 0
        
        
        for i, s in enumerate(predict_y_linear_regressor_length_size):
            predict_y_linear_regressor_length_size[i] = round(predict_y_linear_regressor_length_size[i], 1)
            ab = list(math.modf(predict_y_linear_regressor_length_size[i]))
            if ab[0] > 0.5 and ab[0] < 0.7:
                ab[0] = 0.5
            elif ab[0] > 0.3 and ab[0] < 0.5:
                ab[0] = 0.5
            elif ab[0] > 0.7:
                ab[0] = 1
            else:
                ab[0] = 0
            predict_y_linear_regressor_length_size[i] = ab[0]+ab[1]     
        
        
        for i, s in enumerate(predict_y_linear_regressor_width_size):
            predict_y_linear_regressor_width_size[i] = int(predict_y_linear_regressor_width_size[i])
        
        # Writing the Output to the file system to a submit_file.csv file
        df = pd.DataFrame({'shoe_style':predict_y_linear_regressor_shoe_style, 'length_size': predict_y_linear_regressor_length_size, 'width_size': predict_y_linear_regressor_width_size})
        df.to_csv("submit_file.csv", index = False, encoding='utf-8')
        return "Executed Succesfullly!!"

if __name__== "__main__":
    Model_Score_obj = Model()
    Model_Score_obj.train('shoe_purchases.csv', retrain = False)
    Model_Score_obj.predict('Test.csv')

