import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

TEST_SIZE = 0.3


def main():
    # Check command-line arguments

    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python shopping.py data")

    name = "data"

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(name)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print("Accuracy: " + str(accuracy_score(y_test,predictions)))

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv(os.getcwd() + "\\shopping.csv")
    le = LabelEncoder()

    df['Month'] = le.fit_transform(df['Month'])
    df['VisitorType'] = le.fit_transform(df['VisitorType'])
    df['Weekend'] = le.fit_transform(df['Weekend'])
    df['Revenue'] = le.fit_transform(df['Revenue'])
    labels = (df.iloc[:, -1:]).values.tolist()
    evidence = (df.iloc[:, :-1]).values.tolist()
    return evidence, labels


def train_model(X_train, Y_train):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=4)
    model = svm.SVC()
    model = model.fit(X_train, np.ravel(Y_train))
    return model


def evaluate(y_actual, y_pred):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    TPR = 1-(tp/(tp+fn))
    TNR = (tn/(tn+fp))
    return TPR, TNR


if __name__ == "__main__":
    main()
