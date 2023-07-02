"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity     :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project     :  1 - Predict Customer Churn with Clean Code
  Author      :  Rakan Yamani
  Date        :  10 June 2023
  Description :  This is a model that contains function of churn customer analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import logging
import pandas as pd
import churn_library as cls
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    filename='./logs/churn_library.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s - %(message)s',
    datefmt="%m/%d/%y %I:%M:%S %p")


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    # validat function is working:
    try:
        path = "./data/bank_data.csv"
        df = cls.import_data(path)
        logging.info(
            f"[Testing] SUCCESS: file '{path}' has been loaded successfully")
    except FileNotFoundError:
        logging.error(f"[Testing] ERROR: file '{path}' was not found")

    # validat argument is with expected type:
    try:
        assert isinstance(df, pd.DataFrame)
        logging.info(
            f"[Testing] SUCCESS: method has a valid DataFrame argument {type(df)}")
    except AssertionError:
        logging.error(
            f"[Testing] ERROR: method has an invalid argument {type(df)}")

    # validat dataframe content:
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError:
        logging.error("[Testing] SUCCESS: file forms a currect dataframe")

    # validat predection data is prepared:
    try:
        assert "Churn" in df.columns.values
        logging.info("[Testing] SUCCESS: Churn column is found")
    except AssertionError as error:
        logging.error(
            "[Testing] ERROR: dataframe doesn't contain Churn column")

    return None


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    # validat function is working:
    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        logging.info("[Testing] SUCCESS: the function is working as expected")
    except KeyError:
        logging.error(
            "[Testing] ERROR: The dataframe is missing some columns for the eda")

    categorical_columns = [
        'Attrition_Flag',
        'Card_Category',
        'Gender',
        'Education_Level',
        'Income_Category',
        'Marital_Status']

    quantitative_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']

    df_columns = set(df.columns)
    all_columns = set(categorical_columns + quantitative_columns)
    remaining_columns = all_columns - all_columns.intersection(df_columns)

# validat defined columns:
    try:
        assert all_columns <= df_columns
        logging.info("[Testing] SUCCESS: All provided columns names are valid")
    except AssertionError:
        logging.error(
            f"[Testing] ERROR: Some columns names are missing - {remaining_columns}")

    # validat EDA Images folder:
    try:
        assert os.path.exists("./images/eda")
        logging.info("[Testing] SUCCESS: EDA images folder existed")
    except BaseException:
        logging.info("[Testing] ERROR: EDA images folder does not exist")

    # validat EDA Results folder:
    try:
        assert os.path.exists("./images/results")
        logging.info(
            "[Testing perform_eda] SUCCESS: EDA resulted folder existed")
    except BaseException:
        logging.info(
            "[Testing perform_eda] ERROR: EDA resulted folder does not exist")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    # validat function is working:
    try:
        df = cls.import_data("./data/bank_data.csv")
        categorical_columns = [
            'Attrition_Flag',
            'Card_Category',
            'Gender',
            'Education_Level',
            'Income_Category',
            'Marital_Status']
        response = [col + '_Churn' for col in categorical_columns]
        df = cls.encoder_helper(df, categorical_columns, response)
        logging.info("[Testing] SUCCESS: the function is working as expected")
    except KeyError:
        logging.error("[Testing] ERROR: encodig function is not working")

# validat all provided column names in category_lst are strings:
    try:
        assert all(isinstance(col_name, str)
                   for col_name in categorical_columns)
        logging.info(
            '[Testing] SUCCESS: provided column names in "categorical_columns" were String')

    except BaseException:
        logging.error(
            '[Testing] ERROR: provided column names in "categorical_columns" were not String')

# validat all provided column names in response are strings:
    try:
        assert all(isinstance(col_name, str) for col_name in response)
        logging.info(
            '[Testing] SUCCESS: provided column names in "response" were of type String')
    except BaseException:
        logging.info(
            '[Testing] ERROR: provided column names in "response" were of type not String')

# validat the list of original categorical column names is matching the
    try:
        assert len(categorical_columns) == len(response)
        logging.info(
            '[Testing] SUCCESS: "categorical_columns" and "response" lists were matched')
    except BaseException:
        logging.error(
            '[Testing] ERROR: "categorical_columns" and "response" lists were not matched')


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    # validat function is working:
    try:
        df = cls.import_data("./data/bank_data.csv")
        categorical_columns = [
            'Attrition_Flag',
            'Card_Category',
            'Gender',
            'Education_Level',
            'Income_Category',
            'Marital_Status']
        df = cls.encoder_helper(df, categorical_columns, '')
        drop_columns = categorical_columns + ['CLIENTNUM', 'Churn']
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df, drop_columns)
        logging.info('[Testing] SUCCESS: the function is working as expected')
    except AssertionError:
        logging.error('[Testing] ERROR: the function is not working')

    # validat dropped columns:
    try:
        assert not set(drop_columns) & set(x_train.columns)
        assert not set(drop_columns) & set(x_test.columns)
        logging.info('[Testing] SUCCESS: required culomns were dropped')
    except AssertionError:
        logging.error('[Testing] ERROR: required culomns were not dropped')

# validate y values ('Churn') was correctly placed
    try:
        assert "Churn" not in set(x_train.columns)
        assert "Churn" not in set(x_test.columns)
        assert "Churn" in y_train.name
        assert "Churn" in y_test.name
        logging.info("[Testing] SUCCESS: 'Churn' was correctly placed")
    except AssertionError:
        logging.error("[Testing] ERROR: 'Churn' was misplaced")

    return x_train, x_test, y_train, y_test


def test_train_models(model_rf, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    # validat function is working:
    try:
        cls.train_models(model_rf, x_train, x_test, y_train, y_test)
        logging.info('[Testing] SUCCESS: the function is working as expected')

    except AssertionError:
        logging.error('[Testing] ERROR: the function is not working')


if __name__ == "__main__":

    logging.info("[### ---| Testing import_data method |--- ###]")
    test_import("")
    logging.info("[### Completed: import_data method  ###]")

    logging.info("[### ---| Testing test_eda method |--- ###]")
    test_eda("")
    logging.info("[### Completed: test_eda method  ###]")

    logging.info("[### ---| Testing test_encoder_helper method |--- ###]")
    test_encoder_helper("")
    logging.info("[### Completed: test_encoder_helper method ###]")

    logging.info(
        "[### ---| Testing test_perform_feature_engineering method |--- ###]")
    x_train, x_test, y_train, y_test = test_perform_feature_engineering("")
    logging.info(
        "[### Completed: test_perform_feature_engineering method ###]")

    logging.info("[### ---| Testing test_train_models method |--- ###]")
    model_rf = RandomForestClassifier(
        max_depth=13,
        criterion='entropy',
        max_features='auto',
        random_state=42)
    test_train_models(model_rf, x_train, x_test, y_train, y_test)
    logging.info("[### Completed: test_train_models method ###]")
