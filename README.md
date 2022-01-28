## disaster_response_pipeline
Data Engineering Project for Udacity Data Science Nano Degree


## Project Overview
Apply data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

Data set contains real messages that were sent during disaster events. A machine learning pipeline is built to categorize these events so can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

This project demonstrates software skills, including ability to create data pipelines and write clean, organized code.


## Project Components
Three components for this project;

# 1. ETL Pipeline
In Python script, process_data.py, data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

# 2. ML Pipeline
In Python script, train_classifier.py, machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

# 3. Flask Web App
Flask web app (utilisaing html, css and javascript) that:

- Provides data visualizations using Plotly in the web app


## Installations

Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly


## Instructions

Run the following commands in the project's root directory to set up your database and model.
        
To run ETL pipeline that cleans data and stores in database
        
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv 
        data/disaster_response_message.db
        
To run ML pipeline that trains classifier and saves python
        
        models/train_classifier.py data/disaster_response_message.db models/disaster_response_message_model.pkl

Run the following command in the app's directory to run your web app. 
        
        python run.py

Go to http://0.0.0.0:3001/ This may work better if you change the host address to 127.0.0.1 in the Run.py file and use the URL http://127.0.0.1:3001/


## Results

The flask App runs and displays the classes associated with the message than the user inputs. Plotly graphs of the message dataset are displayed on the webpage.
Licensing, Authors, Acknowledgements

Creative Commons Licence

This work is licensed under a Creative Commons Attribution 4.0 International License

The Disaster Response Message Dataset is provided by Figure Eight

Graphs by [Plotly] (https://plotly.com/)
