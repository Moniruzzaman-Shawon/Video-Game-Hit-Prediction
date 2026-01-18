import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import mlflow
import mlflow.sklearn


def main():
    # load data
    df = pd.read_csv("vgsales.csv")
    print("Dataset shape:", df.shape)

    # drop useless columns
    df.drop(columns=["Rank", "Name"], inplace=True)

    # convert year to numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # target: hit if global sales >= 1 million
    df["Hit"] = (df["Global_Sales"] >= 1.0).astype(int)

    # remove leakage column
    df.drop(columns=["Global_Sales"], inplace=True)

    # split X and y
    X = df.drop("Hit", axis=1)
    y = df["Hit"]

    # find numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    print("Numeric features:", list(numeric_features))
    print("Categorical features:", list(categorical_features))

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # preprocessing for numbers
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # preprocessing for categories
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # combine preprocessors
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ])

    # full pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # train model
    rf_pipeline.fit(X_train, y_train)

    # cross validation
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print("CV Mean Accuracy:", cv_scores.mean())
    print("CV Std:", cv_scores.std())

    # grid search tuning
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print("Best CV Accuracy:", grid_search.best_score_)
    print("Best Parameters:", grid_search.best_params_)

    # best model
    best_model = grid_search.best_estimator_

    # test evaluation
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", test_acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # mlflow logging (local)
    tracking_path = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file:///{tracking_path.replace(os.sep, '/')}")
    mlflow.set_experiment("VideoGame_Hit_Prediction")

    with mlflow.start_run(run_name="RF_Final_Model"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_accuracy", grid_search.best_score_)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model_pipeline")

    print("MLflow logging done!")

    # save model for gradio
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Saved model: best_model.pkl")
    print("Model size (KB):", round(os.path.getsize("best_model.pkl") / 1024, 2))


if __name__ == "__main__":
    main()
