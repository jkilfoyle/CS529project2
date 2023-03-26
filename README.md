# CS529_project2




------------------
**Naive Bayes**


The file naive_bayes.py corresponds to the Naive Bayes classifier. It can be executed with no arguments, and will calculate
accuracy using B=1/V by default. It assumes all csv files needed are in the directory above it. It will also print out the top 100 mutual information words.

Optional Calculations:
By default, it calculates accuracy and runs only on the validation data. To make predictions on the test data, a few line will need to be commented, and a few uncommented. These lines are indicated in comments.

The naive bayes model can be used with a range of beta values, at the bottom these are block commented out, but can be uncommented to run in a loop and plot accuracy across the range of beta values.
