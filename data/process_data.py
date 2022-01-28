import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath)->pd.DataFrame:
    """
    Loads Message and Category Data Files
    Args:
    messages_filepath (string) pathname of messages file
    categories_filepath (string)  pathname of categories file
    Returns:
    df (pd.DataFrame) DataFrame containing data loaded.
    """

    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages,categories, on='id')


def clean_data(df)->pd.DataFrame:
    """
    Cleanup Message Data
    Args:
    df (pd.DataFrame) containing data to process
    
    Returns:
    df (pd.DataFrame) containing cleaned up data
    """
    # Fill original Nulls
    df['original'].fillna(value='-',inplace=True)
    # Extract Categories
    categories = df['categories'].str.split(pat=';',n=-1,expand=True)

    #
    row = categories.iloc[0]
    category_colnames = row.loc[:].apply(lambda x : x[:-2])
    categories.columns = category_colnames

    # Expand Category Data
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='raise')
        # Fix value to one
        categories[column]=np.where(categories[column]>1,1,categories[column])

    df.drop(['categories'],axis=1,inplace=True)

    # Concatenate message and expanded category dtat
    df = pd.concat([df, categories],axis=1)

    # Remove Duplicates
    num_dup1=df.duplicated(subset='message').sum()
    print('Number of duplicate messages',num_dup1)
    df.drop_duplicates(subset='message', inplace=True)
    num_dup2=df.duplicated(subset='message').sum()
    print('Number of duplicate messages removed',num_dup1-num_dup2)
    
    # Check for Nulls
    num_nulls=np.sum(df.isna().sum())
    print("Number of Nulls in the dataset =", num_nulls)

    return df


def save_data(df, database_filename):
    """
    Save data to SQLLite file
    Args:
    df (pd.DataFrame)
    database_filename
    Returns:
    """
    tableName='Message_Categories'
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(tableName, engine, index=False,if_exists='replace' )
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
