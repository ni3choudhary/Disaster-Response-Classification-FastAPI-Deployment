import sys
import pandas as pd
from sqlalchemy import create_engine
 
def load_data(messages_filepath, categories_filepath):
    """
    load and merge the messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    merged_df = pd.merge(messages, categories, on = "id")
    return merged_df

def clean_data(merged_df):

    # split the category labels into separate columns
    categories = merged_df.categories.str.split(';',expand=True)

    # extract category labels
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    for col in categories:
        # set each value to be the last character of the string
        categories[col] = categories[col].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[col] = categories[col].astype("int")

    # clip values to be either a 0 or 1
    categories = categories.clip(0, 1)

    # remove columns with only one value
    one_value_columns = [
        col for col in categories.columns if len(categories[col].unique()) == 1
    ]

    categories.drop(one_value_columns, axis=1, inplace=True)

    # drop original categories column and add encoded features to dataframe
    merged_df.drop("categories", axis=1, inplace = True)
    df = pd.concat([merged_df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    # Save DataFrame to sqlite database
    connect_str = f"sqlite:///{database_filename}"
    engine = create_engine(connect_str)

    df.to_sql('messages', engine, index = False, if_exists = 'replace')

def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )

        merged_df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(merged_df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python ETL-Pipeline/data_processing.py "
            "data/disaster_messages.csv data/disaster_categories.csv "
            "database/DisasterResponse.db"
        )


if __name__ == '__main__':
    main()