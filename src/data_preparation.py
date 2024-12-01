
import pandas as pd
import numpy as np
import os


def import_join_csv(list_paths, id_column):
    """
    Given a list of paths to CSVs, where each file has a column with the exact same name, merge all.
    After merging, transform column names to lowercase.

    Args:
        list_paths (_type_): _description_
    """
    # Make paths relative to the current working directory
    list_paths = [os.path.abspath(os.path.join(os.getcwd(), path)) for path in list_paths]

    # import first path
    df_total = pd.read_csv(list_paths[0])

    # load subsequent
    for path in list_paths[1:]:
        df = pd.read_csv(path)

        # check for common join columns
        common_columns = list(set(df_total.columns).intersection(df.columns))
        if not id_column in common_columns:
            raise ValueError(f"Id_column {id_column} not common between {list_paths[0]} and {path}")

        # join dfs
        df_total = df_total.merge(df, on=id_column, how="inner")

    # drop common id_column
    df_total.drop(columns=id_column, inplace=True)

    # convert column names to lowercase
    df_total.columns = [col.lower() for col in df_total.columns]

    return df_total

list_of_paths = ["../data/general_data.csv", "../data/manager_survey_data.csv","../data/employee_survey_data.csv"]

def drop_unique_value(dataframe):
    # drop columns with just 1 value
    for feature in dataframe.columns:
        if dataframe[feature].nunique() == 1:
            dataframe.drop(columns=feature, inplace=True)

    return dataframe

def convert_to_object(dataframe, additional_columns=None):
    #convert columns with 15 values or less to object (discrete)
    for feature in dataframe.select_dtypes(np.number).columns:
        if dataframe[feature].nunique() <= 15:
            dataframe[feature] = dataframe[feature].astype("object")
    
    if additional_columns:
        dataframe[additional_columns] = dataframe[additional_columns].astype("object")

    return dataframe

def regroup_categories_attrition(dataframe):
    dataframe.replace({"educationfield": {"Human Resources": "Other"}}, inplace=True)
    dataframe["percentsalaryhike"] = np.where(dataframe["percentsalaryhike"] >= 23, "23 or more", dataframe["percentsalaryhike"])
    dataframe["yearssincelastpromotion"] = np.where(dataframe["yearssincelastpromotion"] >= 8, "8 or more", dataframe["yearssincelastpromotion"])
    dataframe["yearswithcurrmanager"] = np.where(dataframe["yearswithcurrmanager"] >= 10, "10 or more", dataframe["yearswithcurrmanager"])

    return dataframe

def remove_incorrect_values(dataframe):
    # convert to nan impossible condition for "yearsatcompany" and "totalworkingyears" 
    condition = dataframe["yearsatcompany"] > dataframe["totalworkingyears"]
    dataframe.loc[condition,["yearsatcompany","totalworkingyears"]] = np.nan

    return dataframe


def load_and_clean(list_of_paths, id_column):
    repl_dict = {"attrition":{"Yes": 1, "No": 0}}

    # load
    employee_attrition = import_join_csv(list_of_paths, id_column)


    # convert target to number
    employee_attrition = employee_attrition.replace(repl_dict)

    # drop duplicates
    employee_attrition.drop_duplicates(inplace=True)

    # drop nan in target column
    employee_attrition.dropna(subset="attrition",inplace=True)


    # drop variables with 1 category
    employee_attrition = drop_unique_value(employee_attrition)

    # convert discrete numerical to object
    additional_columns = ["yearssincelastpromotion","yearswithcurrmanager"]
    employee_attrition = convert_to_object(employee_attrition, additional_columns)

    # intentional - 0 is the same as NaN here
    employee_attrition.loc[(employee_attrition["numcompaniesworked"] == 0),"numcompaniesworked"] = "0"
    print("ksks")


    # regroup categories
    employee_attrition = regroup_categories_attrition(employee_attrition)


    # remove "error" outliers
    employee_attrition = remove_incorrect_values(employee_attrition)


    # ensure target is numeric
    employee_attrition["attrition"] = employee_attrition["attrition"].astype("Int64")

    # ensure all object types are categorical
    employee_attrition.select_dtypes(["object","category"]) = (employee_attrition.select_dtypes(["object","category"])
                                                                .astype("str").astype("object"))


    return employee_attrition