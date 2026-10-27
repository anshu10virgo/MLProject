import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path: str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("Splitting training and test array")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], 
                train_arr[:,-1],
                test_arr[:,:-1], 
                test_arr[:,-1]
            )

            models = {
                "Random Forest Regression": RandomForestRegressor(),
                "Gradient Boosting Regression": GradientBoostingRegressor(),
                "Ada Boosting Regression": AdaBoostRegressor(),
                "XG Boosting Regression": XGBRegressor(),
                "Cat Boosting Regression": CatBoostRegressor(),
                "Decision Tree Regression": DecisionTreeRegressor(),
                "K-Neighbours Regression": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge()
            }


            params={
                "Random Forest Regression":{
                                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                                    
                                    # 'max_features':['sqrt','log2',None],
                                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regression":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Ada Boosting Regression":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XG Boosting Regression":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boosting Regression":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Decision Tree Regression": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "K-Neighbours Regression":{
                    'n_neighbors':np.arange(1, 21),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Linear Regression":{},
                "Lasso Regression":{
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                },
                "Ridge Regression":{
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }                
            }
            logging.info("Evaluating Models")
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test,
                                               models=models, params=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model score name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                               ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info(f"Best found model on training and testing dataset {best_model}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
