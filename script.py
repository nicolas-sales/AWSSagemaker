
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import os
import pandas as pd
import numpy as np


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf  

if __name__=="__main__":

    print("Extracting arguments")
    print()

    parser=argparse.ArgumentParser()

    # Hyperparameter
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--train-file", type=str, default="train.csv")  
    parser.add_argument("--test-file", type=str, default="test.csv")   

    args, _ = parser.parse_known_args()

    print("sklearn version :", sklearn.__version__)
    print("joblib version :", joblib.__version__)
    print("Reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("Building training and testing datasets")
    print()
    X_train= train_df.drop(columns=['price_range'])
    y_train = train_df["price_range"]

    X_test= test_df.drop(columns=['price_range'])
    y_test = test_df["price_range"]

    print("Data shape")
    print()
    print("Training data")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("Testing data")
    print(X_test.shape)
    print(y_test.shape)
    print()

    print("Training RandomForest Model")
    model=RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        verbose=2,
        n_jobs=1
    )

    model.fit(X_train,y_train)

    print()

    model_path=os.path.join(args.model_dir,"model.joblib")
    os.makedirs(args.model_dir, exist_ok=True)  # s’assurer que le dossier existe
    joblib.dump(model,model_path)

    print("Model saved at" + model_path)

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_rep = classification_report(y_test,y_pred_test)

    print()
    print("Metrics results for testing data")
    print()
    print("Model accuracy:", test_acc)
    print("Testing report:", test_rep)
