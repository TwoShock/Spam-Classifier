import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

direc = 'emails/'#change this to your local path for email dataset

def getEmailFileDir():
    """
    @function: makes a list of paths to emails
    @return: list of email paths
    """
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    return emails

def getEmailLabels():
    """
    @function: labels emails as spam or not based on if it contains the string ham in filename
    @return: list of email labels 1's and 0's 1 is spam 0 is not spam
    """
    labels = []
    emails = getEmailFileDir()
    for email in emails:
        try:
            f = open(email)
            if "ham" in email:
                labels.append(0)
            if "spam" in email:
                labels.append(1)
        except:
            print(email,"Error labeling")
    return labels

def createEmailCorpus():
    """
    @functtion: get all words used in all emails
    @return: word list
    """
    emails = getEmailFileDir()
    corpus = []
    for email in emails:
        try:
            f = open(email)
            readStream = f.read()
            corpus.append(readStream) 
        except:
            print(email,"Error in adding email to corpus")
    return corpus

def createDataFrame():
    """
    @function: creates a dataframe containing occurence of word count of all words across
    all emails per email. Contains the Label column which tells if the email being viewed
    is spam or not
    @return: datafram containg the wordcount and label fields
    """
    labels = getEmailLabels()
    corpus = createEmailCorpus()
    vectorizer = CountVectorizer()
    
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names())
    df['Labels'] = labels

    return df

def main():
    df = createDataFrame()
    X = df.loc[:,df.columns != 'Labels']
    y = df['Labels']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_train)
    print(classifier.predict('I am'))
    print(classification_report(y_train ,pred ))
    print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
    print()
    print('Accuracy: ', accuracy_score(y_train,pred))

if(__name__ == "__main__"):
    main()
