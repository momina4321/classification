# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import data as d
import utils as model
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    data_path="F:\FYP\ecg_data_with_labels-5_cleaned-6.csv"
    df = pd.read_csv(data_path, index_col=None, header=0)
    train_df, test_df = train_test_split(df, test_size=0.2)  # 80% data for training and 20% for testing
    train_df, val_df = train_test_split(train_df, test_size=0.25)  # 25% of the data will be used for validation and 75% for training
    print(df)
    return train_df,test_df,val_df



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_df,test_df,val_df=load_data()
    model.create_supervised_task(train_df,val_df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
