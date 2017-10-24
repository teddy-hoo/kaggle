# encoding: utf-8

import tensorflow as tf
import pandas as pd
from data_provider import (
    train_x,
    train_y,
    eval_x,
    eval_y,
    test_x,
    output_2_file
)


x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


batch_size = 100


def next_batch(data_set, count=None):
    if not count:
        return data_set
    total = len(data_set)
    begin = batch_size * count % total
    end = (begin + batch_size) % total
    return data_set[begin:end]


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(5000):
        batch_xs = next_batch(train_x, i)
        batch_ys = next_batch(train_y, i)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: eval_x, y_: eval_y}))

    predict = tf.argmax(y.eval({x: test_x}), 1).eval()

    output_2_file(predict)
