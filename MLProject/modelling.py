import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="namadataset_preprocessing")
args = parser.parse_args()

mlflow.sklearn.autolog()

X_train = pd.read_csv(f"{args.data_path}/X_train.csv")
X_test = pd.read_csv(f"{args.data_path}/X_test.csv")
y_train = pd.read_csv(f"{args.data_path}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{args.data_path}/y_test.csv").values.ravel()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Test accuracy:", acc)
