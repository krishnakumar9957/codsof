import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preprocessing
# Load your dataset (replace 'data.csv' with your dataset)
data = pd.read_csv('Churn_Modelling.csv')
print(data.head())


# Handle missing data if any
data.fillna(0, inplace=True)  # Replace missing values with zeros

# Encode categorical features
# Encode categorical features
encoder = LabelEncoder()
categorical_cols = ['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']  # Add other categorical columns as needed
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])


# Split the data into training and testing sets
X = data.drop(columns=['Exited'])
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Selection (You may need more advanced methods)
# For simplicity, we'll use all available features for now

# Step 3: Model Building and Training
# Logistic Regression
lr_model = LogisticRegression(max_iter=1000) #You can set the higher value if its needed
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

lr_accuracy, lr_report = evaluate_model(lr_model, X_test, y_test)
rf_accuracy, rf_report = evaluate_model(rf_model, X_test, y_test)
gb_accuracy, gb_report = evaluate_model(gb_model, X_test, y_test)

print("Logistic Regression:")
print(f"Accuracy: {lr_accuracy}")
print(lr_report)

print("Random Forest:")
print(f"Accuracy: {rf_accuracy}")
print(rf_report)

print("Gradient Boosting:")
print(f"Accuracy: {gb_accuracy}")
print(gb_report)

sns.countplot(data=data, x='Exited')
plt.title('Churn Distribution')
plt.show()
numeric_cols = ['Age', 'Balance', 'EstimatedSalary']
for col in numeric_cols:
    sns.histplot(data=data, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
sns.pairplot(data=data, hue='Exited')
plt.title('Pair Plot')
plt.show()
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()
# Step 5: Predict Churn for New Data
# You can use the chosen model (e.g., gb_model) for making predictions on new data.
# Example:
# new_data = pd.read_csv('new_data.csv')  # Load new data
# new_data_encoded = encode_categorical_features(new_data)  # Encode categorical features
# churn_predictions = gb_model.predict(new_data_encoded)
