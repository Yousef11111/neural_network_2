"""Practice code for RNN use in tensorflow"""
import numpy as np
import tensorflow as tf


INSTANCES_COUNT = 100
TIME_STEPS = 100
FEATURES_SIZE = 10
LABELS_SIZE = 20
LSTM_SIZE = 50
NUM_STEPS = 5
assert NUM_STEPS <= TIME_STEPS


class PracticeRNN(object):
    """A class for building a RNN"""

    def __init__(self):
        self.inputs = tf.placeholder(
            tf.float32,
            [None, None, FEATURES_SIZE])
        self.labels = tf.placeholder(
            tf.float32,
            [None, None, LABELS_SIZE])
        self.seq_len = tf.placeholder(
            tf.float32,
            [None])
        batch_size = tf.shape(self.inputs)[0]
        time_steps = tf.shape(self.inputs)[1]
        cell = tf.contrib.rnn.LSTMCell(
            num_units=LSTM_SIZE,
            state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(
            cell,
            self.inputs,
            sequence_length=self.seq_len,
            initial_state=initial_state)
        self.converter = tf.Variable(
            tf.random_uniform(
                [cell.output_size, LABELS_SIZE],
                -1.0,
                1.0))
        reshaped = tf.reshape(
            outputs,
            [
                batch_size * time_steps,
                cell.output_size])
        self.predictions = tf.reshape(
            tf.matmul(reshaped, self.converter),
            [
                batch_size,
                time_steps,
                LABELS_SIZE])
        self.loss = tf.reduce_mean(
            tf.square(tf.subtract(self.labels, self.predictions)))
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = optimizer.minimize(self.loss)


# pylint:disable-msg=no-member
def generate_data():
    """Generates data"""
    return \
        np.random.rand(INSTANCES_COUNT, TIME_STEPS, FEATURES_SIZE), \
        np.random.rand(INSTANCES_COUNT, TIME_STEPS, LABELS_SIZE)


def _run():
    """Trains RNN"""
    features, labels = generate_data()
    with tf.Graph().as_default(), tf.Session() as sess:
        net = PracticeRNN()
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {
            net.inputs: features,
            net.seq_len: [TIME_STEPS]*features.shape[0],
            net.labels: labels}
        for _ in range(10):
            loss, _ = sess.run([net.loss, net.train], feed_dict=feed_dict)
            print(loss)


if __name__ == '__main__':
    _run()