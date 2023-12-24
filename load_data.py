import os
import numpy as np
import scipy.io
import re
from preprocessing import extract_bands, mat_extractor

def load_data(data_file_dir):
    data_files = os.listdir(data_file_dir)
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []
    train_data = []
    train_labels = []

    for data_file in data_files:
        if not re.search(".*\.mat", data_file):
            continue

        info = re.findall('B0([0-9])0([0-9])([TE])', data_file)
        try:
            subject = "subject" + info[0][0]
            session = "session" + info[0][1]
            data_type = info[0][2]  # T or E
            filename = data_file_dir + "/" + data_file

            if info[0][1] == '3':
                print(f"{subject}:{session}:{data_type}:  loading------- ")
                xx, yy = mat_extractor(filename, data_type)
                print(f"{subject}:{session}:{data_type}: xx{xx.shape}, yy{yy.shape} ")
                val_data.append(xx)
                val_labels.append(yy)
                print(f"val_data: {len(val_data)}------- ")
                print(f"val_labels: {len(val_labels)}------- ")

            elif info[0][1] in ['4', '5']:
                print(f"{subject}:{session}:{data_type}:  loading------- ")
                xx, yy = mat_extractor(filename, data_type)
                print(f"{subject}:{session}:{data_type}: xx{xx.shape}, yy{yy.shape} ")
                test_data.append(xx)
                test_labels.append(yy)
                print(f"test_data: {len(test_data)}------- ")
                print(f"test_labels: {len(test_labels)}------- ")

            elif info[0][1] in ['1', '2']:
                print(f"{subject}:{session}:{data_type}:  loading------- ")
                xx, yy = mat_extractor(filename, data_type)
                print(f"{subject}:{session}:{data_type}: xx{xx.shape}, yy{yy.shape} ")
                train_data.append(xx)
                train_labels.append(yy)
                print(f"train_data: {len(train_data)}------- ")
                print(f"train_labels: {len(train_labels)}------- ")

        except Exception as e:
            print(f"Exception: {e} !!!! ")

    # Convert lists to 3D arrays
    val_data = np.concatenate(val_data, axis=0)
    print(f"validation dataset shape: {val_data.shape} !!!")
    val_labels = np.concatenate(val_labels, axis=0)
    print(f"validation labels shape: {val_labels.shape} !!!")

    test_data = np.concatenate(test_data, axis=0)
    print(f"test dataset shape: {test_data.shape} !!!")
    test_labels = np.concatenate(test_labels, axis=0)
    print(f"test labels shape: {test_labels.shape} !!!")

    train_data = np.concatenate(train_data, axis=0)
    print(f"train dataset shape: {train_data.shape} !!!")
    train_labels = np.concatenate(train_labels, axis=0)
    print(f"train labels shape: {train_labels.shape} !!!")

    np.save(f'{data_file_dir}/val_data.npy', val_data)
    np.save(f'{data_file_dir}/val_labels.npy', val_labels)
    np.save(f'{data_file_dir}/test_data.npy', test_data)
    np.save(f'{data_file_dir}/test_labels.npy', test_labels)
    np.save(f'{data_file_dir}/train_data.npy', train_data)
    np.save(f'{data_file_dir}/train_labels.npy', train_labels)

    print("Data Loading Done!!!")
