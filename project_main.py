import pandas as pd
import numpy as np
from collections import defaultdict

def train_naive_bayes(training_data, vocab_file):
    v = len(pd.read_csv(vocab_file, header=None))
    beta = 1 / v
    alpha = 1 + beta
    p_yk = training_data.iloc[:, -1].value_counts(normalize=True).to_dict()

    count_xi_in_yk = training_data.groupby(training_data.columns[-1]).sum()
    words_in_yk = count_xi_in_yk.sum(axis=1)
    
    p_xi_yk = (count_xi_in_yk + (alpha - 1)).div(words_in_yk + (alpha - 1) * v, axis=0)

    return p_xi_yk, p_yk

def classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk):
    log_pxiyk = np.log2(classifier_pxiyk)
    log_pyk = np.log2(pd.Series(classifier_pyk))

    #Uncomment next line if using validation set
    test_data_no_labels = test_data.drop(test_data.columns[-1], axis=1)
    log_probabilities = test_data_no_labels.dot(log_pxiyk.T) + log_pyk
    predictions = log_probabilities.idxmax(axis=1).values

    return predictions

def calc_accuracy(test_data, predictions):
    correct = (test_data.iloc[:, -1].values == predictions).sum()
    n = len(predictions)
    return correct / n if n > 0 else 0

# Specify path to training data
file = "../training.csv"

# Use pandas to read the CSV file into a dataframe
df = pd.read_csv(file)

# Use 2/3 of the data for training
training_data = df.sample(frac=0.66)

# Use 1/3 for testing
test_data = df.drop(training_data.index)

print("data loaded 1")
classifier_pxiyk, classifier_pyk = train_naive_bayes(training_data, "../vocabulary.txt")
print("classifiers 1")
predictions = classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk)
print("predictions 1")
print(predictions)
print(len(predictions))
print(len(test_data.index))
print("Accuracy=", calc_accuracy(test_data, predictions))
