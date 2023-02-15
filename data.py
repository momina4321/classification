
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def preparing_data():
    # first we need to load the dataset
    path = r'C:\Users\Nadeem\PycharmProjects\FYP\mitbih_database'

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    annotation_files = glob.glob(os.path.join(path, "*.txt"))
    csv_li = []
    for files in csv_files:
        df = pd.read_csv(files, index_col=None, header=0)
        csv_li.append(df)

    csv_data = pd.concat(csv_li, axis=0, ignore_index=True)  # concatinating the dataset

    classes = ['N', 'A', 'R', 'L', 'V', '/', 'a', '!', 'F', 'Q']

    n_classes = len(classes)  # number of classes
    features = ['MLII', 'V5']

    # we will use one shot learning which requires small number of training data to classify new data
    # train and test datasets
    # train_files=["101",'106','109','114','116','201','203','205','207','215','230']
    # validation_files=['112','208','209','122','220','223','118','119','124','115','108']
    # test_files =["100", "103", "105", "111", "113", '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

    Y = []  # labels
    samples = []  # if the sample of .txt matches that of .csv then append it to samples array and append the corresponding label to labels array
    # annotations=[]

    for files in annotation_files:

        with open(files) as f:
            annotationData = f.readlines()  # read the file
        for d in range(1, len(annotationData)):
            splitted = annotationData[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted)  # Time Clipping
            sampleId = int(next(splitted))  # Sample ID
            arrhythmia_type = next(splitted)  # Type
            Y.append(arrhythmia_type)
            samples.append(sampleId)
            # annotations.append(files)

    labels = pd.DataFrame(Y, columns=["'Type'"])

    # we have 'csv_data' as concatenated dataframe

    series = csv_data.loc[samples, ["'sample #'","'MLII'","'V5'"]]
    data = pd.DataFrame(series)
    data = data.reset_index(drop=True)
    data = pd.merge(data, labels, left_index=True, right_index=True)  # preparing final data
    data = data[data["'Type'"].isin(classes)]

    print(data)

    # split the data we have prepared into training, validation and test data

    train_df, test_df = train_test_split(data, test_size=0.2)  # 80% data for training and 20% for testing
    train_df, val_df = train_test_split(train_df,
                                        test_size=0.25)  # 25% of the data will be used for validation and 75% for training


