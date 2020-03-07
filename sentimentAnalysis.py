import re
import nltk
import pprint
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
nltk.download('averaged_perceptron_tagger')

#consolidated regex expressions into extracting text beginning, between, or ending with ' quote
pattern = re.compile(r"(?<![a-zA-Z])'(\D*?)'(?![a-zA-Z])")

#Input: file location
#Output: document is converted to tuples of (sentence, 0 or 1) 0 for negative and 1 for positive sentiment
def load_corpus(corpus_path):
    file = open(corpus_path, "r")
    tuples = []
    for line in file:
        values = line.split('\t')
        snippet = values[0]
        num = values[1].strip('\n')
        tup = (snippet, num)
        tuples.append(tup)
    return tuples

#Input: (sentence, value) tuples
#Output: Tokenized words and wrapped "words" that match the specific regex pattern
def tokenize(snippet):
    tokens = []
    quotes = []
    #findall matches for the regular expression
    match = re.findall(pattern, snippet)
    if(match != None):
        quotes.append(match)

    #grab all initial tokens
    for word in snippet.split():
        tokens.append(word)

    #if there are quotes in the snippet, add surrounding quotes ""
    if(len(quotes[0])>0):
        #grab all quotes
        quoteList = quotes[0]
        #if theres more than one quote
        if(len(quoteList) > 1):
            #handle all quotes
            newSnip = ' '.join(word for word in tokens)
            for quote in quoteList:
                newSnip = re.sub("'" + quote + "'", " ' " + quote + " ' ", newSnip)
                newSnip = re.sub("'" + quote, " ' " + quote, newSnip)
            #new token list
            tokens = []
            for word in newSnip.split():
                tokens.append(word)

        #if theres only one quote
        else:
            #if theres more than one word in the quote
            first = " ' " + quoteList[0].strip("'")
            #if theres only one word in the quote
            firstboth = " ' " + quoteList[0].strip("'") + " ' "

            #handle each case
            for i, w in enumerate(tokens):
                if (w == ("'" + quoteList[0].split()[0])):
                    tokens.remove(w)
                    tokens.insert(i, first)

                if (w == ("'" + quoteList[0].split()[0] + "'")):
                    tokens.remove(w)
                    tokens.insert(i, firstboth)

    #get the string of all tokens
    rv = ' '.join(word for word in tokens)
    #return the split of the string
    return rv.split()

#Test cases
print(tokenize('hello \'world'))
print(tokenize('hello world\''))
print(tokenize('don\'t hello'))
print(tokenize('\'hello\' world'))
print(tokenize('\'em world'))

#Input: tokenized words
#Output: for tokens in quotes, we label 'EDIT_' before each token. Return updated tokens
def tag_edits(tokenized_snippet):
    rvTokens = []
    #index of [
    #initialized at -1
    starti = -1
    for i,token in enumerate(tokenized_snippet):
        #if found [ then update index
        if '[' in token:
            starti = i

        #if not found [ then append token
        if (starti == -1):
            rvTokens.append(token)

        #if found ] then update index and handle
        if ']' in token:
            nToken = 'EDIT_' + token.strip('[]')
            if(nToken != 'EDIT_'):
                rvTokens.append(nToken)
            starti = -1

        #if we have seen [ and still have not seen ]
        if(starti != -1):
            newToken = 'EDIT_' + token.strip('[]')
            if(newToken != 'EDIT_'):
                rvTokens.append(newToken)
                
    return rvTokens


#NEGATIION
negation = re.compile(r"(n't|not|\bno\b|never)")
#keywords to end negation in sentences
endneg = ["but","however","nonetheless", "nevertheless", ".", "?", "!"]

#Input: tokenized words
#Output: label NOT_ before each token that appears after a negation until endneg tokens occur
def tag_negation(tokenized_snippet):
    rvTokens = []
    editWords = []

    #seperate out EDIT tags
    for i, token in enumerate(tokenized_snippet):
        if('EDIT_' not in token):
            rvTokens.append(token)
        else:
            editWords.append(token.strip('EDIT_'))
            rvTokens.append(token.strip('EDIT_'))

    #get POS tags
    initTuples = nltk.pos_tag(rvTokens)
    rvTuples = []
    for tup in initTuples:
        word = tup[0]
        pos = tup[1]
        #add back the EDIT tag
        if(word in editWords):
            rvTuples.append(('EDIT_'+ word, pos))
        else:
            rvTuples.append(tup)

    #find NOT tuples
    notTuples = []
    foundNeg = False
    foundEnd = False
    for i, tup in enumerate(rvTuples):
        word = tup[0]
        pos = tup[1]
        #match negation regex
        if(re.search(negation, word) != None):
            if(i != len(rvTuples)- 1):
                nextTup = rvTuples[i+1]
                nextWord = nextTup[0]
                #check cornercase for 'not only'
                if(nextWord != 'only'):
                    foundNeg = True
                    foundEnd = False
                    notTuples.append((word, pos))
                    continue

        if (pos in ['JJR', 'RBR']):
            foundEnd = True

        #if found negation word but haven't found end neg then tag NOT
        if(foundNeg == True and foundEnd == False):
            notTuples.append(('NOT_' + word, pos))
        else:
            #just append tokens
            notTuples.append((word, pos))
        #found end neg word
        if(word[-1] in endneg):
            foundEnd = True

    return notTuples


#Returns feature vector for processed training data
def get_features(preprocessed_snippet):
    #initialize feature vector
    feature_vect = np.zeros(len(feature_dict.keys()))
    for tuple in preprocessed_snippet:
        w = tuple[0]
        for i, key in enumerate(feature_dict.keys()):
            if(w == key):
                #found match, increment count
                feature_vect[i] += 1
    return feature_vect


#Normalize Matrix
def normalize(X):
    arrMax = np.max(X, axis=0)
    rvX = np.zeros((len(X), len(arrMax)))
    for i, col in enumerate(X.T):
        values = col
        rvX[:,i] = np.nan_to_num((values - min(values)) / (max(values) - min(values)))
    return rvX


############Training data############
listOfTuples = load_corpus("train.txt")
print('load_corpus() result')
print(listOfTuples[:50])

#Call functions to process training data
allTuples = []
for i in listOfTuples:
    tokenList = tokenize(i[0])
    taggedTokens = tag_edits(tokenList)
    taggedNots = tag_negation(taggedTokens)
    for tag in taggedNots:
        allTuples.append(tag)

#Create feature dict
ind = 0
feature_dict = dict()
for tuple in allTuples:
    word = tuple[0]
    pos = tuple[1]
    #create vocabulary with unique words
    if('EDIT_' not in word) and (word not in feature_dict.keys()):
        feature_dict[word] = ind
        ind += 1

#Create training matrix X and label vector Y
X_train = np.zeros((len(listOfTuples), len(feature_dict.keys())))
Y_train = np.zeros(len(listOfTuples))
for ind, i in enumerate(listOfTuples):
    tokenList = tokenize(i[0])
    label = i[1]
    taggedTokens = tag_edits(tokenList)
    taggedNots = tag_negation(taggedTokens)
    vector = get_features(taggedNots)

    #populate matrix and label vector
    X_train[ind] = vector
    Y_train[ind] = label

#normalize X train
X_train = normalize(X_train)


###########Evaluation measures: Precision, Recall, and Fmeasure
from sklearn.metrics import precision_score, recall_score, f1_score
def evaluate_predictions(Y_pred, Y_true):
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    fmeasure = f1_score(Y_true, Y_pred)
    print('Evaluate prediction scores:')
    print((precision, recall, fmeasure))
    return (precision, recall, fmeasure)

#Simple Test Case
evaluate_predictions([0,1,0,1], [0,1,1,0])


#############Test data###############
testTuples = load_corpus("test.txt")

#Create test matrix X_test and label vector Y_true
X_test = np.zeros((len(testTuples), len(feature_dict.keys())))
Y_true = np.zeros(len(testTuples))
for ind, tup in enumerate(testTuples):
    tokenList = tokenize(tup[0])
    label = tup[1]
    taggedTokens = tag_edits(tokenList)
    taggedNots = tag_negation(taggedTokens)
    vector = get_features(taggedNots)
    #populate test matrix and label vector
    X_test[ind] = vector
    Y_true[ind] = label

#normalize test matrix
X_testnorm = normalize(X_test)

############Logistic Regression Model##########
from sklearn.linear_model import LogisticRegression
lrModel = LogisticRegression()
lrModel.fit(X_train, Y_train)
#Generate predictions on test set
Y_pred = lrModel.predict(X_testnorm)
evaluate_predictions(Y_pred, Y_true)


#Display top k features for a trained model
def top_features(model, k):
    #sort by second val in tuple
    def sortSecond(val):
        return abs(val[1])
    coef = model.coef_[0]

    listTups = []
    for i, val in enumerate(coef):
        listTups.append((i, val))

    #sort in descending by abs weight value
    listTups.sort(key = sortSecond, reverse=True)

    #return tuples
    rvTups = []
    for tup in listTups:
        index = tup[0]
        for key, val in feature_dict.items():
            #found match
            if(val == index):
                rvTups.append((key, tup[1]))

    #return to the kth element
    return rvTups[:k]

print(top_features(lrModel, 10))
