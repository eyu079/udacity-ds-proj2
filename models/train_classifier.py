import sys

import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///EstherDatabaseName.db')
    df = pd.read_sql_table("DisasterTable",engine)
    df = df.dropna(how="all", subset=df.columns[4:])
    X = df.message.values
    y = df.iloc[:, 4:].values
    categories = [str(category_name) for category_name in df.iloc[0, 4:].index]

    return X, y, categories

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(KNeighborsClassifier())),
    ])


    return pipeline
        

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
  
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    for i in range(len(category_names)):
       print("Label:", category_names[i])
       print(classification_report(Y_test[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
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


if __name__ == '__main__':
    main()