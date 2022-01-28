import sys
import sklearn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import pickle
import re

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

np.random.seed(42)

def load_data(database_filepath):
    """
    Load the data from the database
    Args:
    database_filepath
    Returns:
    X: pd.Series
    Y: pd.DataFrame
    category names:
    """
    # load data from database
    tableName='Message_Categories' # This really needs to be passed in
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name=tableName, con=engine)

    feature_list=['id', 'message', 'original', 'genre']
    X = df['message']
    Y = df.drop(feature_list, axis=1)


    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenize the text message fields
    
    Args:
    text (string) text to tokenize
    
    Returns:
    List tokenised text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    
    lemmatizer = WordNetLemmatizer()
    # Lemmatize
    tokenised = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
                 
    return tokenised
def get_eval_metrics(actual, predicted, col_names):
    """
    Calculate evaluation metrics for model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: List of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # average{‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
    avg_type='weighted'  # weighted is supposed to take label imbalance into account 
    zero_division_treatment=0 # 0,1,'warn'
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average=avg_type, zero_division=zero_division_treatment)
        recall = recall_score(actual[:, i], predicted[:, i], average=avg_type, zero_division=zero_division_treatment)
        f1 = f1_score(actual[:, i], predicted[:, i], average=avg_type, zero_division=zero_division_treatment)
        
        metrics.append( [accuracy, precision, recall, f1] )
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


# Define performance metric for use in grid search scoring object
def performance_metric(y_true, y_pred)->float:
    """
    
    Calculate median F1 score for all of the output classifiers
    
    Args:
    y_true: array. Array containing actual labels.
    y_pred: array. Array containing predicted labels.
        
    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    average_type='binary'
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i],average='micro')
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score

def build_model():
    """
    Create and Train model 
    """
    pipeline1 = Pipeline ( [
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ] )

    # Create grid search object

    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[100, 150], 
              'clf__estimator__min_samples_split':[2, 5, 10]}

    scorer = make_scorer(performance_metric)
    model = GridSearchCV(pipeline1, param_grid = parameters, scoring = scorer, cv=3, verbose = 10, n_jobs=None)

    return model


def get_eval_metrics(actual, predicted, col_names):
    """
    Calculate evaluation metrics for model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: List of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # average{‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
    avg_type='weighted'  # weighted is supposed to take label imbalance into account 
    zero_division_treatment=0 # 0,1,'warn'
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average=avg_type, zero_division=zero_division_treatment)
        recall = recall_score(actual[:, i], predicted[:, i], average=avg_type, zero_division=zero_division_treatment)
        f1 = f1_score(actual[:, i], predicted[:, i], average=avg_type, zero_division=zero_division_treatment)
        
        metrics.append( [accuracy, precision, recall, f1] )
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Args:
    model(model)  Model to evaluate
    X_test (array)  test feature data
    Y_test (array)  test target data
    category_names(list)  list of category names
    Returns:
    """
    # Use Model to predict labels for test dataset
    Y_pred = model.predict(X_test)
    
    # Calculate metrics
    eval_metrics = get_eval_metrics(np.array(Y_test), Y_pred, category_names)
    # Print metrics
    print(eval_metrics)
    # Print metrics description
    print(eval_metrics.describe())


def save_model(model, model_filepath):
    """
    Save the classifier model to the location specified
    Args:
    model (model) Classifier Model to save
    model_filepath (string)  path to model pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        print("argv =",sys.argv)


if __name__ == '__main__':
    main()
    




