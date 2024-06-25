import json
import logging
from collections import Counter
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
#코드 정리는 나중에

# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

# 타겟 변수 변환
def transform_target(df):
    df = df.copy()
    df.loc[:, 'BPRS_change'] = (df['BPRS_sum'].diff().shift(-1) < 0).astype(int)
    return df


# 패딩 함수
def pad_sequence(id_df, max_length, seq_cols):
    sequence = id_df[seq_cols].values
    num_padding = max_length - len(sequence)
    padding = np.zeros((num_padding, len(seq_cols)))
    padded_sequence = np.vstack([padding, sequence])
    return padded_sequence

# 주별로 모델 입력 데이터를 준비하는 함수 (타겟 컬럼 추가)
def prepare_data_for_model_by_week(df, max_length, seq_cols, target_col):
    results = []
    for id in df['이름'].unique():
        patient_data = df[df['이름'] == id]
        for week in range(1, 5):  # 항상 1주차부터 4주차까지 고려
            if week in patient_data['week'].unique():
                week_data = patient_data[patient_data['week'] == week]
                padded_seq = pad_sequence(week_data, max_length, seq_cols)
                X_week = np.array([padded_seq], dtype=np.float32)
                y_week = week_data[target_col].dropna().iloc[-1]
            else:
                # 주차 데이터가 없는 경우, 패딩으로 채움
                X_week = np.zeros((1, max_length, len(seq_cols)), dtype=np.float32)
                y_week = np.nan  # 실제값이 없는 경우 NaN으로 채움

            # 결과 저장
            results.append({
                'Patient_ID': id,
                'Week': week,
                'X': X_week,
                'y': y_week
            })
    return results

# 데이터를 텐서로 변환하고 DataLoader를 생성
def convert_results_to_tensors(results):
    X_data = np.vstack([result['X'] for result in results])
    y_data = np.array([result['y'] for result in results], dtype=np.float32)
    # NaN 값은 0으로 대체 (모델 학습을 위해)
    y_data = np.nan_to_num(y_data, nan=0.0)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor
 
def num_data (data):
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    return data


# Split data by week function
def split_data_by_week(df, date_col='targetTime'):
    df['week'] = df.groupby('이름')[date_col].transform(lambda x: ((x - x.min()).dt.days // 7) + 1)
    return df

def find_max_sequence_length_by_week(df, seq_cols):
    max_length = 0
    for _, group in df.groupby(['이름', 'week']):
        if len(group) > max_length:
            max_length = len(group)
    return max_length

"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""

def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """
    file_path ='./safer_yong.csv'
    data = pd.read_csv(file_path)
    data = data.replace({'False': '0', 'True': '1'})
    # One-hot encoding for 'place' column
    data = pd.get_dummies(data, columns=['place'])

    # Converting targetTime to datetime
    data['targetTime'] = pd.to_datetime(data['targetTime'])
    seq_cols = ['Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy',
       'Normalized_Eight_Hour_Entropy', 'first_TOTAL_ACCELERATION','Location_Variability',
       'last_TOTAL_ACCELERATION', 'mean_TOTAL_ACCELERATION',
       'median_TOTAL_ACCELERATION', 'max_TOTAL_ACCELERATION',
       'min_TOTAL_ACCELERATION', 'std_TOTAL_ACCELERATION',
       'nunique_TOTAL_ACCELERATION', 'delta_CALORIES', 'first_HEARTBEAT', 'last_HEARTBEAT',
       'mean_HEARTBEAT', 'median_HEARTBEAT', 'max_HEARTBEAT', 'min_HEARTBEAT',
       'std_HEARTBEAT', 'nunique_HEARTBEAT', 'delta_DISTANCE', 'delta_SLEEP',
       'delta_STEP','sex', 'age', 'place_Unknown', 'place_hallway',
       'place_other', 'place_ward']
    data = num_data(data)

    # Apply the function
    data = split_data_by_week(data)
    patient_ids = data['이름'].unique()
    max_length = find_max_sequence_length_by_week(data,seq_cols)
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_data = data[data['이름'].isin(train_ids)]
    test_data = data[data['이름'].isin(test_ids)]
    train_data = transform_target(train_data)
    test_data = transform_target(test_data)

    max_length = find_max_sequence_length_by_week(train_data, seq_cols)

    train_results = prepare_data_for_model_by_week(train_data, max_length, seq_cols, 'BPRS_change')
    test_results = prepare_data_for_model_by_week(test_data, max_length, seq_cols, 'BPRS_change')

    X_train_tensor, y_train_tensor = convert_results_to_tensors(train_results)
    X_test_tensor, y_test_tensor = convert_results_to_tensors(test_results)
 

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # DataLoader for client training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """
    file_path ='./safer_yong.csv'
    data = pd.read_csv(file_path)
    data = data.replace({'False': '0', 'True': '1'})
    # One-hot encoding for 'place' column
    data = pd.get_dummies(data, columns=['place'])

    # Converting targetTime to datetime
    data['targetTime'] = pd.to_datetime(data['targetTime' ])
    seq_cols = ['Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy',
       'Normalized_Eight_Hour_Entropy', 'first_TOTAL_ACCELERATION','Location_Variability',
       'last_TOTAL_ACCELERATION', 'mean_TOTAL_ACCELERATION',
       'median_TOTAL_ACCELERATION', 'max_TOTAL_ACCELERATION',
       'min_TOTAL_ACCELERATION', 'std_TOTAL_ACCELERATION',
       'nunique_TOTAL_ACCELERATION', 'delta_CALORIES', 'first_HEARTBEAT', 'last_HEARTBEAT',
       'mean_HEARTBEAT', 'median_HEARTBEAT', 'max_HEARTBEAT', 'min_HEARTBEAT',
       'std_HEARTBEAT', 'nunique_HEARTBEAT', 'delta_DISTANCE', 'delta_SLEEP',
       'delta_STEP','sex', 'age', 'place_Unknown', 'place_hallway',
       'place_other', 'place_ward']
    
    data = num_data(data)
    # Apply the function
    data = split_data_by_week(data)
    patient_ids = data['이름'].unique()
    max_length = find_max_sequence_length_by_week(data,seq_cols)
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_data = data[data['이름'].isin(train_ids)]
    test_data = data[data['이름'].isin(test_ids)]
    train_data = transform_target(train_data)
    test_data = transform_target(test_data)

    max_length = find_max_sequence_length_by_week(train_data, seq_cols)

    train_results = prepare_data_for_model_by_week(train_data, max_length, seq_cols, 'BPRS_change')
    test_results = prepare_data_for_model_by_week(test_data, max_length, seq_cols, 'BPRS_change')

    
    X_train_tensor, y_train_tensor = convert_results_to_tensors(train_results)
    X_test_tensor, y_test_tensor = convert_results_to_tensors(test_results)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # DataLoader for global model validation
    gl_val_loader = DataLoader(test_dataset, batch_size=batch_size)

    return gl_val_loader