import pandas as pd
from sklearn.model_selection import train_test_split
data_for_valid = pd.read_csv('labeled_data.csv')
hate_data_raw = data_for_valid.loc[data_for_valid['class'] == 0]
offensive_data_raw = data_for_valid.loc[data_for_valid['class'] == 1]
neither_data_raw = data_for_valid.loc[data_for_valid['class'] == 2]



hate_data_for_train, hate_data_for_valid = train_test_split(hate_data_raw, test_size=0.25)
offensive_data_for_train, offensive_data_for_valid = train_test_split(offensive_data_raw, test_size=0.25)
neither_data_for_train, neither_data_for_valid = train_test_split(neither_data_raw, test_size=0.25)
train_data_set=pd.concat([hate_data_for_train,offensive_data_for_train,neither_data_for_train])
valid_data_set=pd.concat([hate_data_for_valid,offensive_data_for_valid,neither_data_for_valid])
#train_data_set.to_csv('datasets/train_data.csv')
#valid_data_set.to_csv('datasets/valid_data.csv')