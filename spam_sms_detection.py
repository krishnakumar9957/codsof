import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Sample data


# Load the dataset from a CSV file
data = pd.read_csv('spam.csv',encoding='latin1')
print(data.columns)


# Create a DataFrame
df = pd.DataFrame(data)
X = df['v2']
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Logistic Regression Classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)

# Support Vector Machine Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
def evaluate_model(classifier, X_test_tfidf, y_test):
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Evaluate Naive Bayes
nb_accuracy, nb_report = evaluate_model(nb_classifier, X_test_tfidf, y_test)
print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Classification Report:\n", nb_report)

# Evaluate Logistic Regression
lr_accuracy, lr_report = evaluate_model(lr_classifier, X_test_tfidf, y_test)
print("\nLogistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression Classification Report:\n", lr_report)

# Evaluate Support Vector Machine
svm_accuracy, svm_report = evaluate_model(svm_classifier, X_test_tfidf, y_test)
print("\nSupport Vector Machine Accuracy:", svm_accuracy)
print("Support Vector Machine Classification Report:\n", svm_report)
