from os import name
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
import joblib


def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['HeartDisease']), df['HeartDisease']
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X.values)
    return X_scale, y, scaler


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, recall, f1


def train():
    df = pd.read_csv("./df_clear.csv")
    X,Y, scaler  = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=Y)
    

    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
            'l1_ratio': [0.001, 0.05, 0.01, 0.2],
            "penalty": ["l1","l2","elasticnet"],
            "loss": ['log_loss', 'hinge', 'modified_huber'],
            "fit_intercept": [False, True],
            }
    
    mlflow.set_experiment("heart_disease_models")
    with mlflow.start_run():
        lr = SGDClassifier(random_state=42)
        clf = GridSearchCV(lr, params, cv = 3, n_jobs = 4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        (accuracy, precision, recall, f1)  = eval_metrics(y_val, y_pred)
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        penalty = best.penalty
        eta0 = best.eta0
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("eta0", eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        with open("lr_heart.pkl", "wb") as file:
            joblib.dump(best, "lr_heart.pkl")
            joblib.dump(scaler, "scaler.pkl")
