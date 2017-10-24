# encoding: utf-8

import pandas as pd

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

x_size, _ = train_data.shape

train_size = int(x_size * 0.9)

train_set = train_data[:train_size]
eval_set = train_data[train_size:]


labels = train_set.pop('label').values
train_x = train_set.as_matrix()
train_y = pd.get_dummies(labels).as_matrix()


labels = eval_set.pop('label').values
eval_x = eval_set.as_matrix()
eval_y = pd.get_dummies(labels).as_matrix()

test_x = test_data.as_matrix()


def output_2_file(predictions):
    output = pd.DataFrame()
    output['ImageId'] = range(1, len(predictions) + 1)
    output['Label'] = predictions
    output.to_csv('./data/submission.csv', index=False)
