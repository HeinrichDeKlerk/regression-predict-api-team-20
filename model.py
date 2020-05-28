"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    train_rider_df = feature_vector_df
    train_rider_df['Temperature'].fillna(value=train_rider_df['Temperature'].mean(),inplace = True)
    train_rider_df.columns = [col.replace(" ","_") for col in train_rider_df.columns]
    X = train_rider_df.drop(['Rider_Id','Placement_-_Day_of_Month','Placement_-_Weekday_(Mo_=_1)','Confirmation_-_Day_of_Month',
                             'Confirmation_-_Weekday_(Mo_=_1)','Pickup_-_Day_of_Month','Vehicle_Type','User_Id','Order_No','Pickup_-_Time',
                             'Pickup_-_Weekday_(Mo_=_1)','Arrival_at_Pickup_-_Time','Confirmation_-_Time','Placement_-_Time',
                             'Arrival_at_Pickup_-_Day_of_Month', 'Arrival_at_Pickup_-_Weekday_(Mo_=_1)'], axis=1)
    
    df_with_dummy_value = pd.get_dummies(X, drop_first=True)
    df_with_dummy_value[['Platform_1','Platform_2','Platform_3']] = pd.get_dummies(df_with_dummy_value['Platform_Type'], drop_first=True)
    df_with_dummy_value = df_with_dummy_value.drop(['Platform_Type'], axis=1)
    # Reorder columns with the dependent variable Time_from_Pickup_to_Arrival the last column
    column_titles = [column for column in df_with_dummy_value.columns if column != 
                     'Time_from_Pickup_to_Arrival'] + ['Time_from_Pickup_to_Arrival']
    df_with_dummy_value = df_with_dummy_value.reindex(columns=column_titles)
    # split data into predictors and response, response does not need to be scaled
    X_independent = df_with_dummy_value.drop('Time_from_Pickup_to_Arrival', axis=1)
    # import scaler method from sklearn
    from sklearn.preprocessing import StandardScaler
    # create scaler object
    scaler = StandardScaler() 
    scaled = scaler.fit_transform(X_independent)
    #convert scaled values into dataframe
    standardised_X = pd.DataFrame(scaled,columns=X_independent.columns)
    predict_vector = standardised_X
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
