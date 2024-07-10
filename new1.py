import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("C:\\Users\\Kshitija Habade\\OneDrive\\Desktop\\nlp mock\\tweets-train.csv")

class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_probs = {}
        self.class_priors = {}
        self.classes = []

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        total_samples = len(y_train)
        for class_ in self.classes:
            class_samples = X_train[y_train == class_]
            self.class_priors[class_] = len(class_samples) / total_samples
            all_words = [word for tweet in class_samples for word in tweet.split()]
            word_counts = pd.Series(all_words).value_counts()
            self.class_word_probs[class_] = (word_counts + 1) / (len(all_words) + len(word_counts))

    def predict(self, X_test):
        predictions = []
        for tweet in X_test:
            tweet_words = tweet.split()
            probs = {class_: np.log(self.class_priors[class_]) +
                             sum(np.log(self.class_word_probs[class_].get(word, 1e-10)) for word in tweet_words)
                     for class_ in self.classes}
            predicted_class = max(probs, key=probs.get)
            predictions.append(predicted_class)
        return predictions

X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['label'], test_size=0.15, random_state=42)

clf = NaiveBayesClassifier()
clf.train(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

keyword = input("Enter the keyword to search: ")

search_results = data[data['tweet'].str.contains(keyword, case=False)]

print("Search Results:\n")
print(search_results)