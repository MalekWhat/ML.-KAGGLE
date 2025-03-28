# -*- coding: utf-8 -*-
"""PriceNumbers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uOG0rorhFXmTE_PgfS5yCzpaNAyvM8QB
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("/home/malek1t/ML.-KAGGLE/predict_car_plates_prices/data/train.csv", index_col=False)
current_data = datetime.now()



def data_transform(date_in_data):
    difference_in_time = str(current_data - datetime.strptime(date_in_data, "%Y-%m-%d %H:%M:%S")).split()[0]
    return int(difference_in_time)
data["date"] = data["date"].apply(data_transform)


data = data.loc[data.groupby("plate")["date"].idxmin()]






data["Triplet_letters"] = 0
data["Double_letters"] = 0
data["Single_letters"] = 0
data["Sixtet_numbers"] = 0
data["Kvintet_numbers"] = 0
data["Quartet_number"] = 0
data["Triplet_numbers"] = 0
data["Doublet_numbers"] = 0
data["Singlet_numbers"] = 0

for index, plate_in in data[["id", "plate"]].iterrows():
    letter_count = {}
    number_count = {}

    for char in plate_in['plate']:
        if char.isdigit():
            number_count[char] = number_count.get(char, 0) + 1
        else:
            letter_count[char] = letter_count.get(char, 0) + 1

    for let, seq in letter_count.items():
        if seq == 3:
            data.loc[index, "Triplet_letters"] += 1
        elif seq == 2:
            data.loc[index, "Double_letters"] += 1
        elif seq == 1:
            data.loc[index, "Single_letters"] += 1

    for num, seq in number_count.items():
        if seq == 6:
            data.loc[index, "Sixtet_numbers"] += 1
        elif seq == 5:
            data.loc[index, "Kvintet_numbers"] += 1
        elif seq == 4:
            data.loc[index, "Quartet_number"] += 1
        elif seq == 3:
            data.loc[index, "Triplet_numbers"] += 1
        elif seq == 2:
            data.loc[index, "Doublet_numbers"] += 1
        elif seq == 1:
            data.loc[index, "Singlet_numbers"] += 1

    data.shape
    letter_count.clear()
    number_count.clear()


data = data.drop(["id", "plate", "date"], axis=1)
data, y = data.drop(["price"], axis=1), data["price"]
#MMS = MinMaxScaler()
#data_transform = MMS.fit_transform(data)
#data = pd.DataFrame(data=data_transform, columns=data.columns)



xtrain, xtest, ytrain, ytest = train_test_split(data, y, test_size = 0.2, shuffle=True)

dtree = DecisionTreeRegressor()

param_grid3 = {
    "criterion":["squared_error", "absolute_error"],
    "max_depth":[6, 12, 20],
    "min_samples_split":[10, 15, 25],
    "min_samples_leaf":[1, 2, 3],
}
grid_search3 = GridSearchCV(dtree, param_grid3, cv=3)
grid_search3.fit(xtrain, ytrain)
print("Best parametrs of Decision Tree", grid_search3.best_params_)
best_Dtree = grid_search3.best_estimator_
print(mean_absolute_percentage_error(ytest, best_Dtree.predict(xtest)))

RFS = RandomForestRegressor()

param_grid2 = {
    "n_estimators": [2, 4, 8, 10, 15],
    "max_depth": [10, 20, 30],
    "min_samples_split":[15, 20, 25],
    "min_samples_leaf":[1, 2, 3]
}
grid_search2 = GridSearchCV(estimator=RFS, param_grid=param_grid2, cv=5)
grid_search2.fit(xtrain, ytrain)
best_rf = grid_search2.best_estimator_
print("Best parametrs of RFS ", grid_search2.best_params_)
print(mean_absolute_percentage_error(ytest, best_rf.predict(xtest)))



ModelXGB = xgb.XGBRegressor()

param_grid = {
    "n_estimators": [250, 300, 350, 400],
    "max_depth": [7, 10, 15],
    "learning_rate": [0.013, 0.015, 0.02, 0.025],
    "subsample": [0.8, 0.9, 0.99],
    "colsample_bytree": [0.33, 0.66, 1.0] 
}
grid_search = GridSearchCV(estimator=ModelXGB, param_grid=param_grid, cv=5)
grid_search.fit(xtrain, ytrain)
best_model = grid_search.best_estimator_
print("The best parametrs OF XGBoost: ", grid_search.best_params_)
print("MAPE: ",mean_absolute_percentage_error(ytest, best_model.predict(xtest)))