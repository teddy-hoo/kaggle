import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


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


def normalize(cell):
    return (cell - cell.min()) / (cell.max() - cell.min())


for df in train, test:
    df['Age'] = normalize(df['Age'])
    df['Fare'] = normalize(df['Fare'])
    df['Sex'] = normalize(df['Sex'])
    df['Pclass'] = normalize(df['Pclass'])
    df['FamilySize'] = normalize(df['FamilySize'])
    df['Embarked'] = normalize(df['Embarked'])
    df['Title'] = normalize(df['Title'])


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

layers = [7, 5, 3, 2]

weights = []
for i in range(len(layers) - 1):
    weights.append(tf.Variable(tf.random_normal([layers[i], layers[i + 1]])))

biases = []
for i in range(1, len(layers)):
    biases.append(tf.Variable(tf.random_normal([layers[i]])))


previous_layer = None
for i in range(len(weights)):
    if previous_layer is None:
        previous_layer = x
    if i == len(weights) - 1:
        previous_layer = tf.nn.softmax(tf.matmul(previous_layer, weights[i]) + biases[i])
    else:
        previous_layer = tf.nn.relu(tf.matmul(previous_layer, weights[i]) + biases[i])
output_layer = previous_layer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

for i in range(len(weights)):
    cost += 0.009 * tf.nn.l2_loss(weights[i] + biases[i])

optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

init = tf.global_variables_initializer()

plt.ion()

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    sess.run(init)
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    eval_x, eval_y = next_batch(None, eval_set)
    train_x, train_y = next_batch(None, train_set)
    train_acc = []
    eval_acc = []
    train_cost = []
    eval_cost = []
    # fg1, ax1 = plt.subplot(1, 1)
    # fg2, ax2 = plt.subplot(1, 1)
    # ax1.set_xlabel('iter')
    # ax1.set_ylabel('acc')
    # ax2.set_xlabel('iter')
    # ax2.set_ylabel('loss')
    for epoch in range(2000):
        avg_cost = 0
        for i in range(6):
            batch_x, batch_y = next_batch(i, train_set)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / 6

        train_acc.append(accuracy.eval({x: train_x, y: train_y}))
        eval_acc.append(accuracy.eval({x: eval_x, y: eval_y}))
        train_cost.append(avg_cost)
        eval_cost.append(sess.run(cost, feed_dict={x: eval_x, y: eval_y}))

        axis = np.arange(0, epoch + 1, 1)

        # ax1.plot(axis, train_acc, 'r')
        # ax1.plot(axis, eval_acc, 'g')
        # fg1.canvas.draw()
        # ax2.plot(axis, train_cost, 'r')
        # ax2.plot(axis, eval_cost, 'g')
        # fg2.canvas.draw()

        if len(axis) % 20 == 0:
            plt.figure(1)
            plt.title('accuracy')
            line_up, = plt.plot(axis, train_acc, 'r', label='train')
            line_down, = plt.plot(axis, eval_acc, 'g', label='eval')
            plt.legend([line_up, line_down], ['train', 'eval'])
            plt.figure(2)
            plt.title('cost')
            line_up, = plt.plot(axis, train_cost, 'r', label='train')
            line_down, = plt.plot(axis, eval_cost, 'g', label='eval')
            plt.legend([line_up, line_down], ['train', 'eval'])
            plt.draw()
            plt.pause(0.0000000000000000000001)

    print("\nTraining complete!")

    # find predictions on val set
    print("Validation Accuracy:", accuracy.eval({x: eval_x, y: eval_y}))
    plt.show()
    passengerIds = test.pop('PassengerId').values
    test_x, _ = next_batch(None, test)
    pred = output_layer.eval({x: test_x})
    predictions = tf.argmax(pred, dimension=1).eval()

    output = pd.DataFrame()
    output['PassengerId'] = passengerIds
    output['Survived'] = predictions
    output.to_csv('kaggle.csv', index=False)

    # axis = np.arange(0, 500, 1)
    # plt.figure(1)
    # plt.title('accuracy')
    # line_up, = plt.plot(axis, train_acc, 'r', label='train')
    # line_down, = plt.plot(axis, eval_acc, 'g', label='eval')
    # plt.legend([line_up, line_down], ['train', 'eval'])
    # plt.figure(2)
    # plt.title('cost')
    # line_up, = plt.plot(axis, train_cost, 'r', label='train')
    # line_down, = plt.plot(axis, eval_cost, 'g', label='eval')
    # plt.legend([line_up, line_down], ['train', 'eval'])
    # # plt.pause(0.0000001)
    # plt.draw()
    # plt.pause(100)
