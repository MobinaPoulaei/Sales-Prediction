# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:00:28 2024

@author: Mobina
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('train.csv', low_memory=0)
store = pd.read_csv('store.csv')
dataset = pd.merge(train, store, how='inner')

# seperate Date columns to Date, Month and Year
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset["Day"] = dataset["Date"].dt.day
dataset["Month"] = dataset["Date"].dt.month
dataset["Year"] = dataset["Date"].dt.year
dataset.drop("Date", axis=1, inplace=True)

dataset["StateHoliday"] = dataset["StateHoliday"].map({"a": "Public Holiday", "b": "Easter Holiday", "c": "Christmas", "0": "No Holiday"})
dataset["StoreType"] = dataset["StoreType"].map({"a":"Type A", "b":"Type B", "c":"Type C", "d":"Type D"})
dataset["PromoInterval"] = dataset["PromoInterval"].map({np.nan:"NOTHING", "Jan,Apr,Jul,Oct":"Jan_to_Oct", "Feb,May,Aug,Nov":"Feb_to_Nov",
                                                           "Mar,Jun,Sept,Dec":"Mar_to_Dec"})
dataset = pd.get_dummies(dataset)
dataset = dataset.fillna(0)

print(dataset[0])

X = dataset.drop(['Sales', 'Customers'], axis=1)
y = dataset['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf, file)