# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Data Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Serialization
import pickle



# Load the data
df = pd.read_csv("insurance_claims.csv")
df.head()
df.isnull().sum()
df.info()
df.describe()

object_columns = df.select_dtypes(include=['object']).columns
object_description = df[object_columns].describe()

df.replace('?', 'UNKNOWN', inplace=True)

col1 = ["insured_sex", "insured_occupation", "insured_relationship", "incident_severity", "property_damage",
        "police_report_available", "collision_type", "policy_state", "insured_education_level", "auto_make",
        "fraud_reported"]

for column in col1:
    unique_values = df[column].unique()
    value_counts = df[column].value_counts()

    print(f"\nColumn: {column}")
    print(f"Number of Unique Values: {len(unique_values)}")

    print("Unique Value Counts:")
    for index, count in value_counts.items():
        print(f"  {index}: {count}")

print(df.dtypes)

# Explore numerical features
numerical_features = ['age', 'auto_year']
for feature in numerical_features:
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

#Boxplot
for feature in numerical_features:
    plt.figure()
    sns.boxplot(df[feature])
    plt.title(f'Distribution of {feature}')
    plt.show()

# Explore categorical features with rotated x-labels
categorical_features = ['insured_sex', 'insured_occupation', 'incident_severity', 'property_damage',
                         'collision_type', 'policy_state', 'insured_education_level', 'auto_make']
for feature in categorical_features:
    plt.figure()
    countplot = sns.countplot(x=feature, data=df, hue='fraud_reported')
    plt.title(f'Distribution of {feature}')

    # Rotate x-labels
    countplot.set_xticklabels(countplot.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()

# Feature engineering
df['vehicle_age'] = 2023 - df['auto_year']
df.drop(['auto_year'], axis=1, inplace=True)



# Columns to label encode
columns_to_encode = ['insured_sex', 'insured_occupation', 'insured_relationship', 'incident_severity',
                      'property_damage', 'police_report_available', 'collision_type', 'insured_education_level',
                      'fraud_reported', 'policy_state', 'auto_make']

# Create a dictionary to store label encoders
label_encoders = {}

# Iterate over columns and create label encoders
for column in columns_to_encode:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

    # Print the key-value pairs
    print(f"\n{column}:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))

# Save the label encoders
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Resampling for class imbalance
fraudulent_claims = df[df['fraud_reported'] == 1]
non_fraudulent_claims = df[df['fraud_reported'] == 0]

oversampled_data = resample(fraudulent_claims, replace=True, n_samples=len(non_fraudulent_claims), random_state=42)
df_resampled = pd.concat([non_fraudulent_claims, oversampled_data])

df_resampled.head(2)

df_resampled.info()

#count of Fraudulant and Non Fraudulant Claims
df_resampled["fraud_reported"].value_counts()

# Original data
original_stats = df.describe()

# Resampled data
resampled_stats = df_resampled.describe()

# Initialize an empty dictionary
comparison_dict = {}

# Iterate over columns
for column in original_stats.columns:
    original_values = original_stats[column].values
    resampled_values = resampled_stats[column].values

    # Add key-value pairs to the dictionary
    comparison_dict[column] = {
        'Original': dict(zip(original_stats.index, original_values)),
        'Resampled': dict(zip(resampled_stats.index, resampled_values))
    }

# Display the comparison in a tabular format
for key, value in comparison_dict.items():
    print(f"\nFeature: {key}")
    comparison_df = pd.DataFrame(value)
    print(comparison_df)

# Data before resampling
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='fraud_reported', data=df)
plt.title('Distribution of Fraud Reported (Before Resampling)')

# Data after resampling
plt.subplot(1, 2, 2)
sns.countplot(x='fraud_reported', data=df_resampled)
plt.title('Distribution of Fraud Reported (After Resampling)')

plt.tight_layout()
plt.show()

# Model training and evaluation
X = df_resampled[['age', 'insured_sex', 'policy_state', 'incident_severity', 'collision_type',
        'property_damage', 'police_report_available', 'auto_make', 'vehicle_age',
        'insured_education_level', 'insured_occupation']]

y = df_resampled['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
training_score = model.score(X_train_scaled, y_train)
training_score

# Model evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display accuracy in percentage
accuracy_percentage = accuracy * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")
print("\n")
print("Confusion Matrix:\n", conf_matrix)
print("\n")
print("Classification Report:\n", classification_rep)

# Create a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# Show the plot
plt.show()


# Assuming your model has a method to predict probabilities
y_probabilities = model.predict_proba(X_test)[:, 1]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)

# Compute area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

coefficients = model.coef_
print("Coefficients:", coefficients)




# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

"""Support Vector Classification"""



# Initialize the Support Vector Classification (SVC) model
svc = SVC(random_state=42)

# Train the model
svc.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svc.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

"""RandomForestClassifier"""


# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = rf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

# Add labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')

# Show the plot
plt.show()



# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train_scaled, y_train)

# Use predict_proba to get probability estimates
probabilities = rf_classifier.predict_proba(X_test)

# Display the results
for i, (true_class, prob_estimates) in enumerate(zip(y_test, probabilities)):
    predicted_class = rf_classifier.predict([X_test_scaled[i]])[0]
    print(f"Instance {i + 1}: True Class = {true_class}, Predicted Class = {predicted_class}, Probability Estimates = {prob_estimates}")

# Save the model
with open('fraud_detection_rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

# Load the model
with open("fraud_detection_rf_model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)


# Initialize the KNN model
knn = KNeighborsClassifier()

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)


# Assuming y_true contains the true labels and y_pred contains the predicted labels
y_true=y_test
# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")

X_train.columns

# Define the order of features
feature_order = X_train.columns

# Sample new data for testing
new_data = pd.DataFrame({
    'age': [25],
    'insured_sex': [1],
    'insured_occupation': [2],
    'incident_severity': [2],
    'property_damage': [2],
    'police_report_available': [2],
    'collision_type': [4],
    'policy_state': [1],
    'insured_education_level': [4],
    'auto_make': [3],
    'vehicle_age': [15]
}, columns=feature_order)

# Transform the new data using the scaler
new_data_scaled = scaler.transform(new_data)

# Predict with the loaded model
predicted_fraud = loaded_model.predict(new_data_scaled)
print("Predicted Fraud:", predicted_fraud)

# Make predictions
predicted_probabilities = rf_classifier.predict_proba(new_data_scaled)

for i, (true_class, prob_estimates) in enumerate(zip(y_test, predicted_probabilities)):
    predicted_class = rf_classifier.predict([X_test_scaled[i]])[0]
    print(f"Instance {i + 1}: True Class = {true_class}, Predicted Class = {predicted_class}, Probability Estimates = {prob_estimates}")

predicted_probabilities

# # Display predicted probabilities for each class
print("Predicted Probabilities:")
for class_name, probability in zip(rf_classifier.classes_, predicted_probabilities[0]):
    print(f"{class_name}: {probability * 100:.2f}%")
