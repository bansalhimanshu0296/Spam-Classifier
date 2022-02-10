# Spam-Classifier

This project was done as a part of CSCI-B-551 Elements of Artificial Intelligence Coursework under Prof. Dr. David Crandall.

## Command to run the program ##

python3 ./SeekTruth.py [training file] [testing file]

## Observation

There are two text files to be analyzed, among which one contains training data and the other contains testing data. Using these files, we build and train a probability model that predicts whether a statement is true or false based on truth and deception statements given in the training dataset. 

## Approach and design decisions

**Abstraction technique:** Bayes net

This is a classical classifier problem that involves classifying data with large datasets. To begin with, we read the training data file and use dictionaries of lists with keys corresponding to objects, labels, and classes, and values representing comments, categories of comments, and distinct categories. Three empty dictionaries have been created to store truthful and deceptive words and their counts. Also, we created a list of stop words such as "between", "but", "they", etc. Next, we assigned each label and comment into respective lists. In the following step, we calculated the prior that any comment would be truthful or deceptive. We iterate through each comment, clean it by removing punctuation and numbers and converting each character to lowercase. By splitting the comments into tokens, we check if the tokens are stop words then we skip that token. Tokens that already exist in the truthful/deceptive bag of words will be increased by one else the word will be added to the deceptive/truthful bag. We are removing words with fewer than 20 count values. Using both truthful and deceptive words, we are calculating the total cumulative frequency and probability of each word in both bags.  After reading the test data, we make a list of each comment. Every comment is iterated again, and punctuation and numbers are removed, along with each character being converted to lowercase. Then splitting comments into tokens . In our calculations, we use the initial probability of a comment being truthful or deceptive as the prior probability. We traverse through each token and check if it is a stop word then we skip otherwise check if it is present in the truthful/deceptive bag of words. If it is present, we are multiplying the probability of words given the comment is truthful. We repeat the same procedure for deceptive as well. We check for the greater probability, append that value to the list and return this list at the end.

## Challanges

Increasing the accuracy of the model was the key challange in this problem. Several techniques were used to overcome this challange, including using log to get the probability and cleaning the data by removing figures, punctuation, etc. As a result, accuracy increased from 70% to 82.75 percent.


