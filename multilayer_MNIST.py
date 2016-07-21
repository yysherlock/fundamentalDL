import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0]) ** 0.5
    weight_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    output = tf.nn.relu(tf.matmul(input, W) + b)
    return output

def inference(x):
    with tf.variable_scope('hidden_1'):
        hidden1 = layer(x, [784, 256], [256])
    with tf.variable_scope('hidden_2'):
        hidden2 = layer(hidden1, [256,256], [256])
    with tf.variable_scope('output'):
        output = layer(hidden2, [256,10], [10])
    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.scalar_summary("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuarcy

learning_rate = 0.01
train_epochs = 1000
batch_size = 100
display_step = 1

with tf.Graph().as_default():
    # add operations for the Graph
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    global_step = tf.Variable(0, name='global_step', trainable=False)
    output = inference(x)
    cost = loss(output, y)
    train_op = training(cost, global_step)
    eval_op = evaluate(output, y)
    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter("mlp_logs/", graph=sess.graph)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        for epoch in range(train_epochs):
            avg_cost = 0.0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                feed_dict = {
                    x : minibatch_x,
                    y : minibatch_y
                }
                sess.run(train_op, feed_dict=feed_dict) # train a step using a minibatch
                minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                avg_cost += minibatch_cost/total_batch

            if epoch % display_step == 0:
                val_feed_dict = {
                    x : mnist.validation.images,
                    y : mnist.validation.labels
                }
                accuarcy = sess.run(eval_op, feed_dict=val_feed_dict) # evaluate on validation dataset
                print("Validation Error:", (1-accuarcy))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, sess.run(global_step))
                saver.save(sess, "mlp_logs/model-checkpoint", global_step=global_step)

        print("optimization finished!")
        # test on test set
        test_feed_dict = {
            x : mnist.test.images,
            y : mnist.test.labels
        }
        accuarcy = sess.run(eval_op, feed_dict=test_feed_dict)
        print("Test Accuarcy:", accuarcy)
