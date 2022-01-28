#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


conda install -c anaconda nltk


# In[2]:


# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.svm import SVC

np.random.seed(42)


# In[3]:


# load data from database
database_filepath = r"C:\Users\diarm\OneDrive\DOCUMENTS\EDUCATION\Data_Science\Udacity data science nanodegree\data engineering\Project_disaster response pipeline\disaster_response_db.db"
engine = create_engine('sqlite:///' + database_filepath)
table_name = database_filepath.replace(".db","") + "_table"
df = pd.read_sql_table(table_name,engine)


# In[4]:


df.head(5000)


# In[5]:


df.describe()


# In[6]:


# Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
# Could have assumed it to be 0. In the absence of information I have gone with majority class.
df['related']=df['related'].map(lambda x: 1 if x == 2 else x)


# In[7]:


# Extract X and y variables from the data for the modelling
X = df['message']
Y = df.iloc[:,4:]


# ### 2. Write a tokenization function to process your text data

# In[8]:


# def tokenize(text):
#     pass

def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


# In[9]:


# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[10]:


# pipeline = 

pipeline = Pipeline ( [
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ] )


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state = 42)

print(X_train.shape)
print(Y_train.shape)


# In[12]:


# Train pipeline model
model1=pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[14]:


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


# In[15]:


# Calculate evaluation metrics for training set
Y_train_pred = pipeline.predict(X_train)
col_names = list(Y.columns.values)

eval_metrics0 = get_eval_metrics(np.array(Y_train), Y_train_pred, col_names)
print(eval_metrics0)


# In[16]:


# Calculate predicted classes for test dataset
Y_test_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
eval_metrics1 = get_eval_metrics(np.array(Y_test), Y_test_pred, col_names)
print(eval_metrics1)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[17]:


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


# In[18]:


# Create grid search object

# commenting out some parameters to reduce runtime with a small number of values each

parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
#               'clf__estimator__n_estimators':[100, 150], 
#               'clf__estimator__min_samples_split':[2, 5, 10]
             }

scorer = make_scorer(performance_metric)
cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, cv=3, verbose = 10, n_jobs=None)

# Find best parameters
np.random.seed(42)
model2 = cv.fit(X_train, Y_train)


# In[19]:


# Print the best parameters in the GridSearch
cv.best_params_


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[33]:


# Calculate evaluation metrics for test set
model2_pred_test = model2.predict(X_test)

eval_metrics2 = get_eval_metrics(np.array(Y_test), model2_pred_test, col_names)

print(eval_metrics2)


# In[34]:


# Get summary stats for tuned model test
eval_metrics2.describe()


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


# Try using DecisionTreeClassifier instead of Random Forest Classifier
pipeline3 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier( DecisionTreeClassifier(splitter='best') ))])


# In[37]:


# List all the parameters for this pipeline
pipeline3.get_params()


# In[38]:


# Create grid search object
# commenting out some parameters to reduce runtime with a small number of values each

parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
#               'clf__estimator__criterion':['gini', 'entropy'], 
#               'clf__estimator__min_samples_leaf':[1, 3]
             }

scorer = make_scorer(performance_metric)
cv = GridSearchCV(pipeline3, param_grid = parameters, scoring = scorer, cv=3, verbose = 10, n_jobs=None)


# In[39]:


# Find best parameters
np.random.seed(42)
model3 = cv.fit(X_train, Y_train)


# In[40]:


# Print the best parameters in the GridSearch
cv.best_params_


# In[41]:


# Calculate evaluation metrics for training set
Y_train_pred = model3.predict(X_train)
col_names = list(Y.columns.values)


# In[42]:


eval_metrics3 = get_eval_metrics(np.array(Y_train), Y_train_pred, col_names)
print(eval_metrics3)


# In[43]:


# Get summary stats for tuned model
eval_metrics3.describe()


# In[44]:


# Calculate evaluation metrics for test set
model3_pred_test = model3.predict(X_test)

eval_metrics4 = get_eval_metrics(np.array(Y_test), model3_pred_test, col_names)

print(eval_metrics4)


# In[45]:


# Get summary stats for model3 test
eval_metrics4.describe()


# ### 9. Export your model as a pickle file

# In[46]:


# Pickle best model
pickle.dump(model3, open('response_message_model.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




