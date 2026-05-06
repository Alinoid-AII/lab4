from os import name
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    df = pd.read_csv("./df_clear.csv")
    X,Y, scaler  = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=Y)
    

    params = {
        'alpha': [0.0008, 0.0009, 0.001, 0.002],
        'l1_ratio': [0.1, 0.3, 0.5, 0.8, 0.9],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'loss': ['log_loss', 'hinge'],
        'fit_intercept': [True, False],
        'eta0': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['optimal', 'invscaling', 'adaptive']}
    
    mlflow.set_experiment("heart_disease_models")

    with mlflow.start_run():
        lr = SGDClassifier(random_state=None, class_weight='balanced')
        clf = RandomizedSearchCV(lr, params, cv=3, n_jobs=4, n_iter=10, random_state=None, scoring='recall')
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        (accuracy, precision, recall, f1)  = eval_metrics(y_val, y_pred)
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("eta0", best.eta0)
        mlflow.log_param("learning_rate", best.learning_rate)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        
        joblib.dump(best, "lr_heart.pkl")
        joblib.dump(scaler, "scaler.pkl")
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        model_info = mlflow.sklearn.log_model(best, "model", signature=signature)
        
        print(model_info.model_uri)

if __name__ == "__main__":
    train()
