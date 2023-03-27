import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

#Naive Bayes Implementation
#CS529 Project 2
#Jeb Kilfoyle
#Abir Islam
#Calculates P(Y), P(X|Y) and does classification according to naive bayes. 
#Also calculates top 100 mutual information words.


#Takes in training data in the form of a pandas frame,
# a vocabulary file directory, and optionally a beta value
def train_naive_bayes(training_data, vocab_file, beta_prior=-1):
    v = len(pd.read_csv(vocab_file, header=None))
    beta = 1 / v
    if not beta_prior == -1: 
        beta = beta_prior
    alpha = 1 + beta
    #Calculate P(Y_k)
    p_yk = training_data.iloc[:, -1].value_counts(normalize=True).to_dict()
    
    #Count of X_i in Y_k across all training data
    count_xi_in_yk = training_data.groupby(training_data.columns[-1]).sum()
    #Count number of words in Y_k total
    words_in_yk = count_xi_in_yk.sum(axis=1)
    
    #Calculate P(X_i|Y_k)
    p_xi_yk = (count_xi_in_yk + (alpha - 1)).div(words_in_yk + (alpha - 1) * v, axis=0)

    return p_xi_yk, p_yk

#Takes in test data in the form of a pandas frame,
#classifier_pxiyk a pandas frame why row j, column i is P(X_i|Y_j)
#classifier_pyk a pandas series with P(Y_k), index is k
def classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk):
    log_pxiyk = np.log2(classifier_pxiyk)
    log_pyk = np.log2(pd.Series(classifier_pyk))

    #Uncomment next lines if using validation set
    test_data_no_labels = test_data.drop(test_data.columns[-1], axis=1)
    log_probabilities = test_data_no_labels.dot(log_pxiyk.T) + log_pyk
    #log_probabilities = test_data.dot(log_pxiyk.T) + log_pyk
    
    predictions = log_probabilities.idxmax(axis=1).values

    return predictions

"""
#Calculates the accuracy of predictions on test_data
def calc_accuracy(test_data, predictions):
    correct = (test_data.iloc[:, -1].values == predictions).sum()
    n = len(predictions)
    return correct / n if n > 0 else 0
"""

#calculates accuracy by summing across confusion matrix
def calc_accuracy(confusion_matrix):
    correct = np.trace(confusion_matrix)
    total = confusion_matrix.sum().sum()
    return correct / total if total > 0 else 0

#Calculates Confusion Matrix
def calc_confusion_matrix(test_data, predictions):
    true_labels = test_data.iloc[:, -1].values
    labels = np.unique(true_labels)
    cm = pd.DataFrame(data=0, columns=labels, index=labels)
    
    for true_label, pred_label in zip(true_labels, predictions):
        cm.at[true_label, pred_label] += 1

    return cm


#Calculates the mutual information between each word X_i and Y
#classifier_pxiyk a pandas frame why row j, column i is P(X_i|Y_j)
#classifier_pyk a pandas series with P(Y_k), index is k
def calc_mi(classifier_pxiyk, classifier_pyk):
    p_yk = pd.Series(classifier_pyk)
    p_xy = classifier_pxiyk.apply(lambda row: row * p_yk[row.name], axis=1)
    p_x = p_xy.sum(axis=0)
    p_x_y_denominator = np.outer(p_yk, p_x)
    mi = p_xy * np.log2(p_xy / p_x_y_denominator)
    mi_overall = mi.sum(axis=0)
    return mi_overall
    
#Helper function for list of top 100 MI
#List of Vocab words
def load_vocabulary(vocab_file):
    with open(vocab_file, 'r') as f:
        vocab = {i: word.strip() for i, word in enumerate(f.readlines())}
    return vocab

#Prints top 100 MI words
def top_n_words_with_mi(mi_series, vocab, n=100):
    top_n_mi = mi_series.nlargest(n)
    top_n_words = [(vocab[word_idx], mi_score) for word_idx, mi_score in top_n_mi.items()]
    return top_n_words

    
# Specify path to training data
file = "training.csv"

# Use pandas to read the CSV file into a dataframe
df = pd.read_csv(file,header=None)
df.drop(columns=df.columns[0], axis=1, inplace=True)
# Use 90% of the data for training
training_data = df.sample(frac=0.90)

# Use 10% for testing
#Swap comments if running validation
test_data = df.drop(training_data.index)
#test_data = pd.read_csv("testing.csv",header=None)
#test_data.drop(columns=test_data.columns[0], axis=1, inplace=True)

#Train classifier on training data
classifier_pxiyk, classifier_pyk = train_naive_bayes(training_data, "vocabulary.txt")
#Make predictions
predictions = classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk)

# Save predictions to a text file
#Uncomment when writing predictions
"""
with open("predictions.txt", "w") as f:
    for i,prediction in enumerate(predictions):
        f.write(str(12001+i) + "," + str(prediction) + "\n")
"""
#Calculate accuracy of predictions

mutual_info = calc_mi(classifier_pxiyk,classifier_pyk)
print(mutual_info)

vocab_file = "vocabulary.txt"
vocab = load_vocabulary(vocab_file)

top_100_words = top_n_words_with_mi(mutual_info, vocab, n=100)

for word, mi_score in top_100_words:
    print(f"{word}: {mi_score}")
    
    
# Calculate confusion matrix
confusion_matrix = calc_confusion_matrix(test_data, predictions)
print("Confusion Matrix:")
print(confusion_matrix)

# Calculate accuracy
accuracy = calc_accuracy(confusion_matrix)
print("Accuracy, beta=1/v: ", accuracy)


#Uncomment to try range of Beta values and plot.
"""
b = [1,0.1,0.01,0.001,0.0001,0.00001]
b.reverse()
accuracy = []
for n in b:
    classifier_pxiyk, classifier_pyk = train_naive_bayes(training_data, "vocabulary.txt", beta_prior=n)
    predictions = classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk)
    # Calculate confusion matrix
    confusion_matrix = calc_confusion_matrix(test_data, predictions)
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Calculate accuracy
    accuracy = calc_accuracy(confusion_matrix)
    print("Accuracy, beta=", n, ": ", accuracy)
# Create a plot
plt.plot(b, accuracy, marker='o')

# Add title and labels to the plot
plt.title('Accuracy vs Betas')
plt.xlabel('Betas')
plt.ylabel('Accuracy')

# Set the x-axis to a log scale
plt.xscale('log')

# Show the plot
plt.show()
"""
