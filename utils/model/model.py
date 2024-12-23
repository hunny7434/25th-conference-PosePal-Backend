# Import required libraries
import os
import pandas as pd
import numpy as np
import pickle
from utils.model.Rocket import RocketTransformerClassifier


############################################################################################################
def pad_with_last_row_new(series, fixed_rows):
    """
    Pad time-series data with the last row to ensure a fixed number of rows.
    Truncate rows if too many, pad with the last row if too few.
    """
    series = np.array(series)
    current_rows, columns = series.shape

    # Truncate if too many rows
    if current_rows > fixed_rows:
        return series[:fixed_rows, :]

    # Pad with the last row if too few rows
    if current_rows < fixed_rows:
        last_row = series[-1, :]  # Get the last row
        padding = np.tile(last_row, (fixed_rows - current_rows, 1))  # Repeat the last row
        return np.vstack((series, padding))  # Add padding rows

    return series


def inference(model_path, dataframes, exercise):
    """
    Load a saved RocketTransformerClassifier model and perform inference on a list of data frames.

    Parameters:
        model_path (str): Path to the saved Rocket model.
        dataframes (list): List of pandas DataFrames containing time-series data.

    Returns:
        list: List of predicted labels for each DataFrame.
    """
    # Load the saved model
    exercise_map = {
        "Side-Lateral-Raise": 11,
        "Lunge" : 15,
    }
    
    print(model_path)

    with open(model_path, "rb") as f:
        rocket_classifier = pickle.load(f)
        
    result = []

    # Extract classifiers from the loaded model
    transformer = rocket_classifier.classifiers_mapping["transformer"]
    scaler = rocket_classifier.classifiers_mapping["scaler"]
    classifier = rocket_classifier.classifiers_mapping["classifier"]

    for df in dataframes:
        # Extract time-series data from DataFrame
        time_series = df.iloc[:, 1:].values  # Exclude non-time-series columns
        time_series = pad_with_last_row_new(time_series, fixed_rows=exercise_map[exercise])  # Ensure fixed number of rows

        # Reshape for inference (Rocket expects 3D array: [samples, time_steps, features])
        x_new = np.expand_dims(time_series, axis=0)  # Add batch dimension

        # Transform and normalize the input data
        x_new_transformed = transformer.transform(x_new)
        x_new_transformed = scaler.transform(x_new_transformed)

        # # Predict class
        # prediction = classifier.predict(x_new_transformed)
        # result.append(prediction[0])
        
        confidence_scores = classifier.decision_function(x_new_transformed)

        # Get top 3 classes based on confidence scores
        top_3_indices = np.argsort(confidence_scores[0])[-3:][::-1]
        top_3_predictions = [classifier.classes_[i] for i in top_3_indices]

        result.append(top_3_predictions)

    return result  # Return the predicted labels


###########################################################################################################
def pad_with_last_row(series, fixed_rows):
    """
    Pad time-series data with the last row to ensure a fixed number of rows.
    Truncate rows if too many, pad with the last row if too few.
    """
    series = np.array(series)
    current_rows, columns = series.shape

    # Truncate if too many rows
    if current_rows > fixed_rows:
        return series[:fixed_rows, :]

    # Pad with the last row if too few rows
    if current_rows < fixed_rows:
        last_row = series[-1, :]  # Get the last row
        padding = np.tile(last_row, (fixed_rows - current_rows, 1))  # Repeat the last row
        return np.vstack((series, padding))  # Add padding rows

    return series


def infer_new_data(model_path, input_dir):
    """
    Load a saved RocketTransformerClassifier model and perform inference on a new CSV file.
    """

    result = []
    # Load the saved model
    with open(model_path, "rb") as f:
        rocket_classifier = pickle.load(f)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)

            # Load CSV file
            df = pd.read_csv(file_path)

            # Extract time-series data
            time_series = df.iloc[:, 1:].values  # Exclude non-time-series columns
            time_series = pad_with_last_row(time_series, fixed_rows=11)  # Ensure fixed number of rows

            # Reshape for inference (Rocket expects 3D array: [samples, time_steps, features])
            x_new = np.expand_dims(time_series, axis=0)  # Add batch dimension

            # Perform inference
            transformer = rocket_classifier.classifiers_mapping["transformer"]
            scaler = rocket_classifier.classifiers_mapping["scaler"]
            classifier = rocket_classifier.classifiers_mapping["classifier"]

            # Transform and normalize the input data
            x_new_transformed = transformer.transform(x_new)
            x_new_transformed = scaler.transform(x_new_transformed)

            # Predict class
            prediction = classifier.predict(x_new_transformed)
            result.append(prediction[0])

    return result  # Return the predicted label


# model_path = "/root/Posepal/25th-conference-PosePal/model/lateralraise_fin.pkl"  # 저장된 모델 경로
# input_dir = "/root/test"  # 새 데이터 CSV 파일이 저장될 경로 - 미리 설정

# predicted_label = infer_new_data(model_path, input_dir)
# # print(f"Predicted label for the new data: {predicted_label}")

# exercise = {'377': '올바른 사이드 레터럴 레이즈 자세', '378': '무릎 반동이 있는 사이드 레터럴 레이즈 자세','379': '어깨를 으쓱하는 사이드 레터럴 레이즈 자세'
#              ,'380': '상완과 전완의 각도 고정이 안 된 사이드 레터럴 레이즈 자세', '381': '손목의 각도가 고정이 되지 않은 사이드 레터럴 레이즈 자세'
#              , '382': '상체 반동이 있는 사이드 레터럴 레이즈 자세'}

# explaination = [exercise[i] for i in predicted_label]

# print(explaination)



