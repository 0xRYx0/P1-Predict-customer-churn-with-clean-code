"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity     :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project     :  1 - Predict Customer Churn with Clean Code
  Author      :  Rakan Yamani
  Date        :  09 June 2023
  Description :  This is a model that contains function of churn customer analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# import libraries
import os
import logging
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap

logging.basicConfig(
    filename='./logs/churn_library.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s - %(message)s',
    datefmt="%m/%d/%y %I:%M:%S %p")

logging.info("SUCCESS: Creating logging file named 'churn_library.log'")


def import_data(pth):
    """
    returns dataframe for the csv found at pth
    input:  pth: a path to the csv
    output: df: pandas dataframe
    """
    # The return can be simply the dataframe without  try and catch close 
    # then return pd.read_csv(pth)
    # Here we will be trying to use a better approach to handle common issues
    # Read the datafile & log both SUCCESS & ERROR attempts:
    try:
        df_original_data = pd.read_csv(pth)
        df_original_data['Churn'] = np.where(df_original_data['Attrition_Flag'] == 'Existing Customer', 0, 1)
        logging.info(f"SUCCESS: file '{pth}' has been loaded successfully")
        return df_original_data

    except FileNotFoundError:
        logging.error(f"ERROR: file '{pth}' was not found")
        return None
    
         


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:  df: pandas dataframe
    output: None
    """
    # preparation - images directory
    eda_path = os.path.abspath(os.getcwd()) + '/images/eda/'

    # histogram plots:
    eda_histplot_details = [{'column': 'Customer_Age',
                             'title': 'Customer Ages Distribution',
                             'file_name': 'customer_age_distribution.png'},
                            {'column': 'Marital_Status',
                             'title': 'Customer Marital Status Distribution',
                             'file_name': 'customer_marital_status_distribution.png'},
                            {'column': 'Total_Trans_Ct',
                             'title': 'Total Transations Count Distribution',
                             'file_name': 'total_trans_count_distribution.png'},
                            {'column': 'Total_Trans_Amt',
                             'title': 'Total Transations Amount Distribution',
                             'file_name': 'total_trans_count_distribution.png'},
                            {'column': 'Avg_Utilization_Ratio',
                             'title': 'Marital Status vs Total Transations Count Distribution',
                             'file_name': 'marital_status_vs_total_trans_count_distribution.png'}]

    for plot in eda_histplot_details:
        fig = plt.figure(figsize=(12, 8))
        sns.histplot(x=df[plot['column']], hue='Churn',
                     kde=True, data=df).set(title=plot['title'])
        fig.savefig(eda_path + 'hist_' + plot['file_name'])
        logging.info(
            f"SUCCESS: EDA operation - ploting histogram plot titled: hist_{plot['file_name']}")

    # count plots:
    eda_countplot_details = [{'column': 'Churn',
                              'title': 'Churn Distribution',
                              'file_name': 'churn_distribution.png'},
                             {'column': 'Education_Level',
                              'title': 'Education Level Distribution ',
                              'file_name': 'education_level_distribution.png'},
                             {'column': 'Income_Category',
                              'title': 'Income Category Distribution',
                              'file_name': 'income_category_distribution.png'},
                             ]

    for plot in eda_countplot_details:
        fig = plt.figure(figsize=(12, 8))
        sns.countplot(x=df[plot['column']]).set_title(plot['title'])
        fig.savefig(eda_path + 'count_' + plot['file_name'])
        logging.info(
            f"SUCCESS: EDA operation - ploting count plot titled: count_{plot['file_name']}")

    # box plots:
    eda_boxplot_details = [{'x': 'Marital_Status',
                            'y': 'Total_Trans_Ct',
                            'title': 'Marital_Status vs TotalTrans Count vs Gender',
                            'file_name': 'marital_status_vs_total_trans_cnt_vs_gender.png'},
                           {'x': 'Marital_Status',
                            'y': 'Total_Trans_Amt',
                            'title': 'Marital_Status vs TotalTrans Amount vs Gender',
                            'file_name': 'marital_status_vs_total_trans_amt_vs_gender.png'}]

    for plot in eda_boxplot_details:
        fig = sns.catplot(
            x=plot['x'],
            y=plot['y'],
            hue='Churn',
            kind='box',
            data=df,
            height=5,
            aspect=2)
        fig.savefig(eda_path + 'box_' + plot['file_name'])
        logging.info(
            f"SUCCESS: EDA operation - Plot: box_{plot['file_name']}")

    fig = sns.jointplot(
        x="Total_Trans_Ct",
        y="Total_Trans_Amt",
        hue="Churn",
        data=df,
        height=10)
    fig.savefig(eda_path + 'joint_total_trans_amt_vs_total_trans_cnt.png')
    logging.info(
        "SUCCESS: EDA operation - Plot: box_joint_total_trans_amt_vs_total_trans_cnt.png")

    plt.figure(figsize=(16, 8))
    mask = np.triu(np.ones_like(df.corr(numeric_only = True), dtype=np.bool_))
    heatmap = sns.heatmap(
        df.corr(numeric_only = True),
        mask=mask,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap='BrBG')
    heatmap.set_title(
        'Triangle Correlation Heatmap',
        fontdict={
            'fontsize': 18},
        pad=16)
    plt.savefig(eda_path + 'heatmap_triangle_corr.png', bbox_inches='tight')
    logging.info(
        "SUCCESS: EDA operation - ploting heatmap plot titled: heatmap_triangle_corr.png")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
                could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for column in category_lst:
        encoded_column = f"{column}_Churn"
        df[encoded_column] = df.groupby(column)['Churn'].transform('mean')

    return df


def perform_feature_engineering(df, drop_cols):
    """
    input:
              df: pandas dataframe
              drop_cols: columns to drop

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    # prepare modeling data (X) and prediction data (y) as required:
    X = df.drop(drop_cols, axis=1)
    print(X.shape)
    print(len(X.columns))
    print(X.columns)
    y = df['Churn']
    logging.info(
        "SUCCESS: Preparing modeling data (X) and prediction data (y)")

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    logging.info(f"x_train data shape: {x_train.shape}")
    logging.info(f"y_train data shape: {y_train.shape}")
    logging.info(f"x_test data shape: {x_test.shape}")
    logging.info(f"y_test data shape: {y_test.shape}")

    return x_train, x_test, y_train, y_test


def classification_report_image(
        model_name,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred
        ):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            model_name: model name to print the right titel
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from logistic regression
            y_test_preds: test predictions from random forest
    output:
             None
    """

    plot_path = os.path.abspath(os.getcwd()) + '/images/results/'

    # classification_report image
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.text(0.01, 0.6, f"{model_name} Train", fontsize=10)
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_pred)), fontsize=10)
    plt.text(0.01, 1.25, f"{model_name} Test", fontsize=10)
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_pred)), fontsize=10)
    plt.savefig(
        plot_path +
        'classification_report_' +
        model_name +
        '.png',
        bbox_inches='tight')


def roc_curve_image(models, x_data, y_data, mode):
    """
    produces roc_curve plot for training and testing results and stores it as image
    in images folder
    input:
            models: model used for generate predictions
            x_data: training data
            y_data: leabel data
            mode: operational mode (train/test)
    output:
             None
    """
    plot_path = os.path.abspath(os.getcwd()) + '/images/results/'

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for model in models:
        metrics.plot_roc_curve(
            model,
            x_data,
            y_data,
            ax=ax,
            alpha=0.8,
            name=type(model).__name__)
    ax.set_title('Receiver Operating Characteristic (ROC) plot for ' + mode)
    plt.savefig(
        plot_path +
        'roc_auc_curve_' +
        mode +
        '.png',
        bbox_inches='tight')


def feature_importance_plot(model, X_data):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    plt_path = os.path.abspath(os.getcwd()) + '/images/results/'

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    model_name = type(model).__name__

    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.title(model_name + "Feature Importance")
    plt.savefig(
        plt_path +
        model_name +
        "_feature_importance.png",
        bbox_inches='tight')


def train_models(model, X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              model: used model in predictions
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    model_name = type(model).__name__
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    auc_score_train = metrics.roc_auc_score(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    auc_score_test = metrics.roc_auc_score(y_test, y_test_pred)
    
    

    logging.info(
        f"SUCCESS: Calculating reuiqred AUC scorse for {model_name}: {str(auc_score_train)} - {str(auc_score_test)}")

    classification_report_image(
        model_name,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred)

    # saving the model
    pickle.dump(
        model,
        open(
            os.path.abspath(
                os.getcwd()) +
            '/models/' +
            model_name +
            '.pkl',
            'wb'))
    logging.info(f"SUCCESS: Exporting trained model '{model_name}.pkl'")


if __name__ == "__main__":

    # run mian script:
    df_data = import_data("./data/bank_data.csv")
    logging.info("SUCCESS: Importing original dataset into dataframe")

    categorical_columns = [
        'Attrition_Flag',
        'Card_Category',
        'Gender',
        'Education_Level',
        'Income_Category',
        'Marital_Status']
    response = [col + '_Churn' for col in categorical_columns]
    df_data = encoder_helper(df_data, categorical_columns, response)
    logging.info("SUCCESS: encoder_helper")

    perform_eda(df_data)
    logging.info("SUCCESS: Performing EDA operations")

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df_data, categorical_columns + ['CLIENTNUM', 'Unnamed: 0', 'Attrition_Flag_Churn','Churn'])
    logging.info("SUCCESS: Performing feature engineering operations")

    model_lr = LogisticRegression(
        solver='lbfgs',
        max_iter=10000,
        random_state=42)
    train_models(model_lr, x_train, x_test, y_train, y_test)
    logging.info("SUCCESS: Trained Logistic Regression Classifier")

    model_rf = RandomForestClassifier(
        max_depth=13,
        criterion='entropy',
        max_features='auto',
        random_state=42)
    train_models(model_rf, x_train, x_test, y_train, y_test)
    logging.info("SUCCESS: Trained Random Forest Classifier")

    roc_curve_image([model_lr, model_rf], x_train, y_train, 'train')
    roc_curve_image([model_lr, model_rf], x_test, y_test, 'test')
    logging.info("SUCCESS: Plotting ROC Curve fro developed models")

    feature_importance_plot(model_rf, x_test)
    logging.info(
        "SUCESS: Computing feature importances for Random Forest model")

# keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
#              'Total_Relationship_Count', 'Months_Inactive_12_mon',
#              'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
#              'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
#              'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
#              'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
#              'Income_Category_Churn', 'Card_Category_Churn']

# Index(['Unnamed: 0', 'Customer_Age', 'Dependent_count', 'Months_on_book',
#        'Total_Relationship_Count', 'Months_Inactive_12_mon',
#        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
#        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
#        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
#        'Churn', 'Attrition_Flag_Churn', 'Card_Category_Churn', 'Gender_Churn',
#        'Education_Level_Churn', 'Income_Category_Churn',
#        'Marital_Status_Churn'],