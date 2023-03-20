#Project main file python 
import pandas as pd
import math
from collections import defaultdict

def train_naive_bayes(training_data,vocab_file):
    v = 0
    #specify path to vocabulary file
    with open(vocab_file, 'r') as file:
        for line in file:
            v += 1
    beta = 1/v
    alpha = 1 + beta
    p_yk = training_data.iloc[:,-1].value_counts(normalize=True).to_dict()
    print("p_yk=", p_yk)
    count_xi_in_yk = defaultdict(int)
    words_in_yk = defaultdict(int)
    for index, row in df.iterrows():
        last_col = row.iloc[-1] # Get the value in the last column
        num_columns = len(row)-1
        for j in range(1,num_columns):
            count_xi_in_yk[(j, last_col)] += row.iloc[j]
            words_in_yk[last_col] += row.iloc[j]
    
    p_xi_yk = {}
    for i in range(1,training_data.shape[1]-1):
        for k in range(1,21):
            p_xi_yk[(i,k)] = count_xi_in_yk[(i,k)]+(alpha-1)
            p_xi_yk[(i,k)] = p_xi_yk[(i,k)]/(words_in_yk[k]+(alpha-1)*v)
    return p_xi_yk,p_yk

def classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk):
    predictions = []
    #For every sample instance
    for index, row in test_data.iterrows():
        #Try all k values
        k_guess = -1
        k_total = float('-inf')
        for key in classifier_pyk:
            t_total = math.log2(classifier_pyk[key])
            for index in range(1,len(row)-1):
                #For every word (x_i)
                t_total += row.iloc[index]*math.log2(classifier_pxiyk[(index,key)])
            if t_total > k_total:
                k_guess = key
                k_total = t_total
        predictions.append(k_guess)
    return predictions

def calc_accuracy(test_data, predictions):
    correct = 0
    n = 0
    for row, prediction in zip(test_data.iterrows(), predictions):
        if prediction == row[1].iloc[-1]:  # row[1] is the actual row data (row[0] is the index)
            correct += 1
        n += 1
    return correct / n if n > 0 else 0
    
# Specify path to training data
file = "../training.csv"

# Use pandas to read the CSV file into a dataframe
#Currently only trains on n=50 rows because my laptop is slow
df = pd.read_csv(file, nrows=50)



#use 2/3 of the data for training
training_data = df.sample(frac = 0.66)
#use 1/3 for testing
test_data = df.drop(training_data.index)

print("data loaded 1")
classifier_pxiyk,classifier_pyk = train_naive_bayes(training_data,"../vocabulary.txt")
print("classifiers 1")
predictions = classify_naive_bayes(test_data, classifier_pxiyk, classifier_pyk)
print("predictions 1")
print (predictions)
print(len(predictions))
print(len(test_data.index))
print("Accuracy=",calc_accuracy(test_data,predictions))
