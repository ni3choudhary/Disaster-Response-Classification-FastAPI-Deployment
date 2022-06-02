# Importing Required Libraries
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from custom_tokenizer import tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, classification_report

from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    """
    Load data from SQLite database and parse into features and target variables
    """

    connect_str = f"sqlite:///{database_filepath}"
    engine = create_engine(connect_str)
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df['message']
    y = df.iloc[:, 4:]
    categories = y.columns.tolist()

    return X, y, categories

def build_model():

    # Parameters to use for grid search. 
    parameters = {
        "vect__tokenizer": [tokenize],
        "vect__max_df": (0.5, 0.75, 1.0, 1.5, 2.0),
        'vect__max_features': (None, 20, 50, 100),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        "clf__estimator__loss" : ("hinge",),
        "clf__estimator__max_iter": (20,),
        "clf__estimator__alpha": (0.00001, 0.00005, 0.000001),
        "clf__estimator__tol" : (0.01,),
        "clf__estimator__n_iter_no_change" : (5,),
        "clf__estimator__penalty": ("l2", "elasticnet"),
        "clf__estimator__max_iter": (100, 400, 500),
    }

    # Machine Learning Pipeline
    # Define a pipeline combining a text feature extractor with a classifier
    pipeline = Pipeline(
        [
            (
                'vect',
                CountVectorizer()

            ),
            ('tfidf', TfidfTransformer()),
            (
                'clf',
                MultiOutputClassifier(SGDClassifier(random_state = 2))
            ),
        ]
    )

    # Find the best parameters for both the feature extraction and the classifier
    ftwo_scorer = make_scorer(fbeta_score, beta=2, average = 'weighted')
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring=ftwo_scorer)
    return grid_search


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    # Print classification report for positive labels
    print(classification_report(y_test, y_pred, target_names= category_names))
    print('Fbeta Score: ', fbeta_score(y_test, y_pred, beta=2, average = 'weighted'))


def save_model(model, model_path):
    '''Pickle the Model'''
    pickle.dump(model, open(model_path , 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))

        X, y, category_names = load_data(database_filepath)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        print('Building Model...')
        model = build_model()

        print('Training Model...')
        model.fit(X_train, y_train)

        print('Evaluating Model...')
        evaluate_model(model, X_test, y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print('Trained Model Saved!')

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            ".\models\classifier.py .\database\DisasterResponse.db .\models\classifier.pkl"
        )


if __name__ == "__main__":
    main()