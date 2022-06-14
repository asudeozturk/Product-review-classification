import nltk
import numpy as np
import pandas as pd

import seaborn as sns #to graph confusion matrix
import matplotlib.pyplot as plt
from itertools import cycle #to color the graph

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def main():
    nltk.download('punkt') #for tokenization

    #READ DATASET
    fields = ["Title", "Review Text", "Rating"]
    df = pd.read_csv('reviews.csv', usecols=fields)

    print("Preprocessing...")

    #CLEAN DATA
    toDelete = []
    for i in range (len(df)):
        entry = df.iloc[i]
        if pd.isna(entry["Title"]) and pd.isna(entry["Review Text"]): #remove if title AND review are n/a
            toDelete.append(i)
        elif pd.isna(entry["Review Text"]):
            df.at[i,"Review Text"] = entry["Title"] #make review = title if review is n/a

    df = df.drop(toDelete)
    df = df.reset_index()

    #LOWERCASE
    for i in range(0,len(df)):
        df.at[i,"Review Text"] = df.at[i,"Review Text"].lower()


    #SPLIT DATA INTO TRAINING AND TEST SETS
    review = df["Review Text"]
    rating = df["Rating"]
    review, reviewTest, rating, ratingTest = train_test_split(review,rating, stratify=rating, test_size=0.2, random_state=42)

    df2 = { "Review Text" : review.tolist(), "Rating": rating.tolist()} #combine review and rating columns of training set
    df2 = pd.DataFrame(df2)

    #BALANCE TRAINING SET SAMPLE
    maxSamples = max(df2.Rating.value_counts()) #find tha max number of samples

    dfClass1 = df2[df2.Rating==1]
    dfClass2 = df2[df2.Rating==2]
    dfClass3 = df2[df2.Rating==3]
    dfClass4 = df2[df2.Rating==4]
    dfClass5 = df2[df2.Rating==5] #majority class

    # Upsample minority class
    df1Up = resample(dfClass1, replace=True, n_samples=maxSamples, random_state=42)
    df2Up = resample(dfClass2, replace=True, n_samples=maxSamples, random_state=42)
    df3Up = resample(dfClass3, replace=True, n_samples=maxSamples, random_state=42)
    df4Up = resample(dfClass4, replace=True, n_samples=maxSamples, random_state=42)
    df5Up = resample(dfClass5, replace=True, n_samples=maxSamples, random_state=42)

    # Combine majority class with upsampled minority class
    dfUpsampled = pd.concat([df1Up, df2Up, df3Up, df4Up, df5Up])


    #BUILD VOCAB VECTOR FOR TRAINING SET
    reviewList = dfUpsampled["Review Text"].tolist()
    ratingList = dfUpsampled["Rating"].tolist()
    vectorizer = CountVectorizer()
    vectorizer.fit(reviewList)         #tokenize and count vocabs
    vocabList = vectorizer.vocabulary_ #contains list of vocab

    #COUNT OCCURENCES OF CLASSES
    classes = dict()
    for r, count in dfUpsampled.Rating.value_counts(sort=False).items():
        classes[int(r)] = count


    #TRAIN MODEL
    print("Training...")
    prior, likelihood = trainNaiveBayes(reviewList, ratingList, vocabList, classes)


    #PREDICT
    print("Testing...")
    reviewTestList = reviewTest.tolist()
    ratingTestList = ratingTest.tolist()

    predictedRating = []

    for i in range(len(reviewTestList)): #test each review in the test set
        rText = reviewTestList[i]
        rRating = ratingTestList[i]
        predictedRating.append( testNaiveBayes(rText, prior, likelihood, classes, vocabList))


    #EVALUATE
    print("Evaluating...")
    showConfusionMatrix(ratingTestList, predictedRating, classes)
    #showROCCurve(ratingTestList, predictedRating, classes) #plot only shows on jpyter notebook

    #DEMO
    while(True):
        text = input("Enter a review: ")
        print("Predicted class:" , testNaiveBayes(text, prior, likelihood, classes, vocabList))


def trainNaiveBayes(reviewList, ratingList, vocabList, classes):
    prior = dict()
    likelihood = dict()
    bigDoc =  dict.fromkeys([c for c in classes], [])

    numDocsD = sum(classes.values()) #total number of reviews (documents)

    for c in classes:
        numDocsC = classes[c]        #number of reviews in class c
        prior[c] = (numDocsC / numDocsD)

        wordCountList = dict()
        for i in range(numDocsD):                    #ratings are grouped by class
            if(ratingList[i] == c ):
                text = reviewList[i]
                bigDoc[c].append(text)

                tokens = nltk.word_tokenize(text)
                for t in tokens:                     #vocab frequency list grouped by class
                    if t in vocabList:
                        if t  not in wordCountList:
                            wordCountList[t] = 1
                        else:
                            wordCountList[t] += 1

        for word in vocabList:
            likelihood[(word,c)] = calculateLikelihood(word, wordCountList, vocabList)

    return prior, likelihood

def testNaiveBayes(review, prior, likelihood, classes, vocabList):
    sums = dict.fromkeys([c for c in classes], [])

    for c in classes:         #calculate probability
        sums[c] = prior[c]
        tokens = nltk.word_tokenize(review)
        for t in tokens:
            if t in vocabList:
                sums[c] = sums[c] + likelihood[t, c]

    maximum = sums[1]
    predicted = 1
    for c, p in sums.items(): #select most probable
        if p > maximum:
            predicted = c
            maximum = p
    return predicted


def calculateLikelihood(word, wcList, vList):
    total = 0
    for v in vList:
        total += (wcList[v]) if v in wcList else  0

    if word in wcList:
        count = wcList[word]
    else:                     #add-1 smoothing
        count = 1
        total += len(vList)

    return count / total

def showConfusionMatrix(ratingTestList, predictedRating, classes):
    confusionMatrix = confusion_matrix(ratingTestList, predictedRating, labels=[c for c in classes])
    ax = sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='d', cbar=False)
    print("Confusion Matrix")
    print(confusionMatrix)

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels([c for c in classes])
    ax.yaxis.set_ticklabels([c for c in classes])

    plt.show()

def showROCCurve(ratingTestList, predictedRating, classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    classNum = len(classes)

    for i in range(1, classNum+1):
        fpr[i], tpr[i], _ = roc_curve(ratingTestList, predictedRating, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])


    colors = cycle(["red", "blue", "orange", "green","purple"])
    for i, color in zip(range(1, classNum+1), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            label="Rating {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC')

if __name__=="__main__":
    main()
