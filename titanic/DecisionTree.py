import pandas as pd
import tensorflow as tf

from warnings import filterwarnings

filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# drop unused field
train.drop("PassengerId", axis=1, inplace=True)
for df in train, test:
    df.drop(["Cabin", "Ticket"], axis=1, inplace=True)

for df in train, test:
    df["Embarked"].fillna("S", inplace=True)
    for feature in "Age", "Fare":
        df[feature].fillna(df[feature].mean(), inplace=True)

for df in train, test:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df.drop(["SibSp", "Parch"], axis=1, inplace=True)

for df in train, test:
    for key, value in zip(("S", "C", "Q"), (0, 1, 2)):
        df.loc[df["Embarked"] == key, "Embarked"] = value
    for key, value in zip(("female", "male"), (0, 1)):
        df.loc[df["Sex"] == key, "Sex"] = value

import re

for df in train, test:
    titles = list()
    for row in df["Name"]:
        surname, title, name = re.split(r"[,.]", row, maxsplit=2)
        titles.append(title.strip())
    df["Title"] = titles
    df.drop("Name", axis=1, inplace=True)

for df in train, test:
    for key, value in zip(("Mr", "Mrs", "Miss", "Master"), list(range(4))):
        df.loc[df["Title"] == key, "Title"] = value

    for title in "Don", "Rev", "Sir":
        df.loc[df["Title"] == title, "Title"] = 0

    for title in "Ms", "Mme":
        df.loc[df["Title"] == title, "Title"] = 1

    for title in "Lady", "Mlle":
        df.loc[df["Title"] == title, "Title"] = 2

    for title in "Dr", "Major", "Col", "Capt", "the Countess", "Jonkheer":
        df.loc[df["Title"] == title, "Title"] = 4

test["Title"][414] = 0

for df in train, test:
    df['Age'] = df['Age'] / (df['Age'].max() - df['Age'].min())
    df['Fare'] = df['Fare'] / (df['Fare'].max() - df['Fare'].min())
    df['Sex'] = df['Sex'] / (df['Sex'].max() - df['Sex'].min())
    df['Pclass'] = df['Pclass'] / (df['Pclass'].max() - df['Pclass'].min())


train_set = train[0:600]
eval_set = train[600:]


def next_batch(i=None, df=None):
    if i is None:
        result = df[:]
    else:
        start = 100 * i
        end = 100 * (i + 1)
        result = df[start:end]
    batch_ys = None
    if 'Survived' in df:
        batch_ys = pd.get_dummies(result.pop('Survived').values).as_matrix()
    batch_xs = result.as_matrix()
    return batch_xs, batch_ys


# define placeholders
x = tf.placeholder(tf.float32, [None, 7])
y = tf.placeholder(tf.float32, [None, 2])


input_size = 7
hidden_layer_1_size = 15
hidden_layer_2_size = 10
hidden_layer_3_size = 5
output_layer_size = 2

weights = {
    'hidden1': tf.Variable(tf.random_normal([7, hidden_layer_1_size])),
    'hidden2': tf.Variable(tf.random_normal([hidden_layer_1_size, hidden_layer_2_size])),
    'hidden3': tf.Variable(tf.random_normal([hidden_layer_2_size, hidden_layer_3_size])),
    'output': tf.Variable(tf.random_normal([hidden_layer_3_size, output_layer_size]))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_layer_1_size])),
    'hidden2': tf.Variable(tf.random_normal([hidden_layer_2_size])),
    'hidden3': tf.Variable(tf.random_normal([hidden_layer_3_size])),
    'output': tf.Variable(tf.random_normal([2]))
}

hidden_layer_1 = tf.nn.relu(tf.matmul(x, weights['hidden1']) + biases['hidden1'])
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights['hidden2']) + biases['hidden2'])
hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, weights['hidden3']) + biases['hidden3'])
output_layer = tf.nn.softmax(tf.matmul(hidden_layer_3, weights['output']) + biases['output'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = tf.train.AdamOptimizer(0.03).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    eval_x, eval_y = next_batch(None, eval_set)
    train_x, train_y = next_batch(None, train_set)
    for epoch in range(1000):
        avg_cost = 0
        for i in range(6):
            batch_x, batch_y = next_batch(i, train_set)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c / 6

        print(
            "Epoch:",
            (epoch + 1), "cost =", "{:.5f}".format(avg_cost),
            "Train Accuracy:", accuracy.eval({x: train_x, y: train_y}),
            "Validation Accuracy:", accuracy.eval({x: eval_x, y: eval_y})
        )

    print("\nTraining complete!")

    # find predictions on val set
    print("Validation Accuracy:", accuracy.eval({x: eval_x, y: eval_y}))

    passengerIds = test.pop('PassengerId').values
    test_x, _ = next_batch(None, test)
    pred = output_layer.eval({x: test_x})
    predictions = tf.argmax(pred, dimension=1).eval()

    output = pd.DataFrame()
    output['PassengerId'] = passengerIds
    output['Survived'] = predictions
    output.to_csv('kaggle.csv', index=False)