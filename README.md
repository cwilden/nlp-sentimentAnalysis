# nlp-sentimentAnalysis
Logistic Regression model to predict sentiment on sentences in a corpus and displays top k features

Input: sentence, 0 or 1. per line in document.

We create tuples of (sentence, 0 or 1). 0 for negative 1 for positive sentiment.

Special case for words that begin, end, or are between a single quote '. We match them with a regex pattern and handle by adding 'EDIT_' token in front of word.

We also have to match negative words using regex and tag these tokens with 'NOT_'. We do this after we encounter a negation token and until we encounter an end negation token.

We then transform these features into vector format from scratch to build X train matrix, Y train label vector, and X test matrix.

We normalize the X_train matrix and X_test matrix seperately.

We chose a Logistic Regression Model to train and predict the test set.

We used the following evaluation scores for predictions: Precision, Recall, and Fmeasure. 

There is also a function at the end to display top K features for a trained model. 
