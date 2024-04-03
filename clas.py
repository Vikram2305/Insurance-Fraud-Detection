import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class InsuranceFraudDetection:
    def __init__(self):
        self.df = pd.read_csv("insurance_claims.csv")

    def preprocess_data(self):
        self.df.replace("?", "UNKNOWN", inplace=True)
        self.df["vehicle_age"] = 2023 - self.df["auto_year"]
        self.df.drop(["auto_year"], axis=1, inplace=True)

    def explore_data(self):
        numerical_features = ["age", "vehicle_age"]
        for feature in numerical_features:
            plt.figure()
            sns.histplot(self.df[feature], kde=True)
            plt.title(f"Distribution of {feature}")
            plt.show()

        for feature in numerical_features:
            plt.figure()
            sns.boxplot(self.df[feature])
            plt.title(f"Distribution of {feature}")
            plt.show()

        categorical_features = [
            "insured_sex",
            "insured_occupation",
            "incident_severity",
            "property_damage",
            "collision_type",
            "policy_state",
            "insured_education_level",
            "auto_make",
        ]
        for feature in categorical_features:
            plt.figure()
            countplot = sns.countplot(
                x=feature, data=self.df, hue="fraud_reported"
            )
            plt.title(f"Distribution of {feature}")
            countplot.set_xticklabels(
                countplot.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            plt.show()

    def label_encode_data(self):
        columns_to_encode = [
            "insured_sex",
            "insured_occupation",
            "insured_relationship",
            "incident_severity",
            "property_damage",
            "police_report_available",
            "collision_type",
            "insured_education_level",
            "fraud_reported",
            "policy_state",
            "auto_make",
        ]
        self.label_encoders = {}
        for column in columns_to_encode:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.label_encoders[column] = le

    def correlation_analysis(self):
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )
        plt.title("Correlation Heatmap")
        plt.show()

    def resample_data(self):
        fraudulent_claims = self.df[self.df["fraud_reported"] == 1]
        non_fraudulent_claims = self.df[self.df["fraud_reported"] == 0]
        oversampled_data = resample(
            fraudulent_claims, replace=True, n_samples=len(non_fraudulent_claims), random_state=42
        )
        self.df_resampled = pd.concat([non_fraudulent_claims, oversampled_data])

    def train_test_split_data(self):
        X = self.df_resampled[
            [
                "age",
                "insured_sex",
                "policy_state",
                "incident_severity",
                "collision_type",
                "property_damage",
                "police_report_available",
                "auto_make",
                "vehicle_age",
                "insured_education_level",
                "insured_occupation",
            ]
        ]
        y = self.df_resampled["fraud_reported"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def train_logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train_scaled, self.y_train)
        self.model = model

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        classification_rep = classification_report(self.y_test, self.y_pred)
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)

        accuracy_percentage = accuracy * 100
        print(f"Accuracy: {accuracy_percentage:.2f}%")
        print("\n")
        print("Confusion Matrix:\n", conf_matrix)
        print("\n")
        print("Classification Report:\n", classification_rep)

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix Heatmap")
        plt.show()

    def save_model(self, filename="fraud_detection_model.pkl"):
        with open(filename, "wb") as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, filename="fraud_detection_model.pkl"):
        with open(filename, "rb") as model_file:
            self.model = pickle.load(model_file)

    def predict_new_data(self, new_data):
        new_data_scaled = StandardScaler().fit_transform(new_data)
        predicted_fraud = self.model.predict(new_data_scaled)
        print("Predicted Fraud:", predicted_fraud)


if __name__ == "__main__":
    fraud_detection = InsuranceFraudDetection()
    fraud_detection.preprocess_data()
    fraud_detection.explore_data()
    fraud_detection.label_encode_data()
    fraud_detection.correlation_analysis()
    fraud_detection.resample_data()
    fraud_detection.train_test_split_data()
    fraud_detection.train_logistic_regression()
    fraud_detection.evaluate_model()
    fraud_detection.save_model("fraud_detection_model.pkl")

    # Example of predicting new data
    new_data = pd.DataFrame(
        {
            "age": [25],
            "insured_sex": [1],
            "insured_occupation": [2],
            "incident_severity": [2],
            "property_damage": [2],
            "police_report_available": [2],
            "collision_type": [4],
            "policy_state": [1],
            "insured_education_level": [4],
            "auto_make": [3],
            "vehicle_age": [15],
        }
    )
    fraud_detection.load_model("fraud_detection_model.pkl")
    fraud_detection.predict_new_data(new_data)
