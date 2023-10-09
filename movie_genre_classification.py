import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the training data
train_path = "C:\\Users\\krish\\Downloads\\archive (3)\\Genre Classification Dataset\\train_data.txt"
train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
print(train_data.describe())

# Load the test data
test_path = "C:\\Users\\krish\\Downloads\\archive (3)\\Genre Classification Dataset\\test_data.txt"
test_data = pd.read_csv(test_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
test_data.head()

# Extract features and labels
X = train_data['Description']
y = train_data['Genre']

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features based on your dataset

# Transform the text data to TF-IDF vectors
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = classifier.predict(X_val)

# Evaluate the performance of the model
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# Create a confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Label the confusion matrix
classes = list(train_data['Genre'].unique())  # Get the unique classes
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')

# Show the plot
plt.show()

# Plot accuracy as a bar chart
genres = list(train_data['Genre'].unique())
accuracy_values = [accuracy_score(y_val[y_val == genre], y_pred[y_val == genre]) for genre in genres]

plt.figure(figsize=(10, 6))
plt.bar(genres, accuracy_values, color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Accuracy')
plt.title('Accuracy by Genre')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set the y-axis limit between 0 and 1
plt.show()



# Evaluate the performance of the model and handle zero division warning
print(classification_report(y_val, y_pred, zero_division=1))

