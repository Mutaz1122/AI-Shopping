


import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
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


def load_data(filename):
    import pandas as pd
    df = pd.read_csv(filename, delimiter = ',')
    #df = pd.read_csv('shopping.csv', delimiter = ',')

    

    e_mapping = {
                   False:0, 
                   True:1,       
                   }
    ed_mapping = {'New_Visitor':0, 
                  'Other':0,
                   'Returning_Visitor':1,

                   }
    edu_mapping = {'Jan':0, 
                   'Feb':1,
                   'Mar':2, 
                   'Apr':3,
                   'May':4, 
                   'June':5,
                   'Jul':6, 
                   'Aug':7,
                   'Sep':8,
                   'Oct':9, 
                   'Nov':10,
                   'Dec':11, 

                   }

    df['Month'] = df['Month'].map(edu_mapping)
    df['VisitorType'] = df['VisitorType'].map(ed_mapping)
    df['Weekend'] = df['Weekend'].map(e_mapping)
    df['Revenue'] = df['Revenue'].map(e_mapping)


   


    evidence=df.iloc[:,:-1].values.tolist()
    labels=df.iloc[:,-1].values.tolist()
    # for ind in df.index:
    #     e=[]    
    #     for col in df2.columns:
    #         e.append(df2[col][ind])
    #     evidence.append(e)
    #     labels.append(df['Revenue'][ind])
    #     e.clear
    
    
    return evidence ,   labels 
    

def train_model(evidence, labels):
    modl=KNeighborsClassifier(n_neighbors=1)
    modl.fit(evidence,labels)
    return modl


def evaluate(labels, predictions):
    predictions=predictions.tolist()
    tp=0
    tn=0
    fp=0
    fn=0
    num=0
    for i in predictions:
        num=num+1
    for i in range(num):
        if predictions[i]==0 and labels[i]==0:
            tn=tn+1
        if predictions[i]==1 and labels[i]==1:
            tp=tp+1
        if predictions[i]==1 and labels[i]==0:
            fp=fp+1    
        if predictions[i]==0 and labels[i]==1:
            fn=fn+1
    sen=tp/(tp+fn)
    spec=tn/(tn+fp) 

    return sen , spec


if __name__ == "__main__":
    main()
