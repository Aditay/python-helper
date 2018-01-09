# from vox_audio import  audiofile_to_input_vector_vox
import pysndfile
from pysndfile import PySndfile
#from pathlib2 import Path
# print(pysndfile.get_sndfile_formats())
import math
import numpy as np
from six.moves import range
import threading
import time
import os
import time
import threading
import tensorflow as tf
#from itertools import izip, izip_longest
from vox_audio import audiofile_to_input_vector_vox, audiofile_to_input_vector_ted

def weight_variable(shape, name):
    return tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial)


def bias_variable(shape, name):
    # initial = tf.constant(0.1, shape=shape)
    # return tf.Variable(initial)
    return tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x, W,stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='VALID')


def max_pool_2x2(x, window, stride):
    return tf.nn.avg_pool(x, ksize=[1, window, 1, 1],
                        strides=[1, stride, 1, 1], padding='VALID')


from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y
flip_gradient = FlipGradientBuilder()



def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # print(value)
    # print(key)
    record_defaults = [['a'], ['b']]
    audio, label = tf.decode_csv(value, record_defaults=record_defaults)
    # print(tf.as_string(audio))
    # print(tf.as_string(label))
    feats, labels = tf.py_func(audiofile_to_input_vector_vox, [audio, label], [tf.float32, tf.int64])
    # print(feats.get_shape().as_list())
    return feats, labels

# def input_pipeline(batch_size = 128, num_epochs=100):
    # filename_queue
batch_size = 128
num_epochs=180
filename_queue1 = tf.train.string_input_producer(['clean_american-v2.csv'], num_epochs=num_epochs, shuffle=True)

filename_queue2 = tf.train.string_input_producer(['clean_british-v2.csv'], num_epochs=num_epochs*10, shuffle=True)

source_batch_x, source_batch_y = read_my_file_format(filename_queue1)
target_batch_x, target_batch_y = read_my_file_format(filename_queue2)
source_batch_x.set_shape([None, 4960 ])
source_batch_y.set_shape([None])
target_batch_x.set_shape([None, 4960])
target_batch_y.set_shape([None])

source_batch_x = tf.reshape(source_batch_x, [-1, 4960, 1, 1])
target_batch_x = tf.reshape(target_batch_x, [-1, 4960, 1, 1])
target_batch_y = tf.reshape(target_batch_y, [-1])
source_batch_y = tf.reshape(source_batch_y, [-1])
    # print(source_x.get_shape().as_list())
min_after_dequeue = 10000
capacity = min_after_dequeue + 3*batch_size
    # print(tf.rank(source_x))
source_x, source_y = tf.train.shuffle_batch([source_batch_x, source_batch_y], batch_size=batch_size, capacity=capacity,
                                     min_after_dequeue=min_after_dequeue, enqueue_many=True, num_threads=3)

target_x, target_y = tf.train.shuffle_batch([target_batch_x, target_batch_y], batch_size=batch_size, capacity=capacity,
                                     min_after_dequeue=min_after_dequeue, enqueue_many=True, num_threads=3)
    # return source_batch_x, source_batch_y, target_batch_x, target_batch_y




# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)

#     for i in range(1200000):
#         example = sess.run([feats_path_source])
    
#     coord.request_stop()
#     coord.join(threads)




batch_size = 128
num_steps = int(2031250*0.6)
class Acoustic_Model(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        # filename_queue = tf.train.string_input_producer(['american_train.tfrecords', 'british_train.tfrecords'], num_epochs=200)
        # reader = tf.TFRecordReader()
        count = tf.Variable(0, name='counter', dtype=tf.float32)
        inc = tf.assign_add(count, 1, name='increment')
        # self.token = tf.Variable(0, name='token', dtype=tf.float32)
        # self.X_source = train_source_x
        # tf.Print(self.X_source, [self.X_source])
        # self.y_source = train_source_y
        # self.X_target = train_target_x
        # self.y_target = train_target_y
        # self.X_source = source_batch_x
        # self.y_source = source_batch_y
        # self.X_target = target_batch_x
        # self.y_target = target_batch_y
        
        # source_x, source_y = custom_runner.queue.dequeue_many(128)
        # target_x, target_y = custom_runner.queue2.dequeue_many(128)
        # # self.X_source = source_x
        # self.token2 = source_x + 1
        # self.y_source = source_batch_y
        # self.X_target = target_batch_x
        # self.y_target = target_batch_y
        # source_x, source_y, target_x, target_y = input_pipeline()
        
        self.X = tf.concat([source_x, target_x], 0)
        self.y = tf.concat([source_y, target_y], 0)
        self.y = tf.cast(self.y, dtype=tf.int32)
        # self.y = tf.cast(self.y, dtype=tf.int32)
        self.y = tf.one_hot(self.y, depth=1677)
        # self.X = tf.reshape(self.X, [None, 31*160, 1, 1])
        # self.domain = tf.concat([self.domain_0, self.domain_1], 0)
        self.train = tf.constant(True, dtype=tf.bool)
        self.l = 2/(1+tf.exp(-10*(inc/num_steps))) -1
        self.lr = 0.005/(tf.pow(1 + 10*(inc/num_steps),0.75))
        # self.l = tf.constant(0.4, dtype=tf.float32)
        # self.lr = tf.constant(0.01, dtype=tf.float32)
        # self.domain_0 = tf.zeros([128], dtype=tf.int32)
        # self.domain_1 = tf.ones([128], dtype=tf.int32)
        self.domain_0 = tf.zeros([128], dtype=tf.int32)
        self.domain_1 = tf.ones([128], dtype=tf.int32)
        rand = tf.random_uniform(shape=[], maxval=1)
        self.domain1 = lambda: tf.concat([self.domain_0, self.domain_1], 0)
        self.domain2 = lambda: tf.concat([self.domain_1, self.domain_0], 0)
        
        self.domain = tf.cond(tf.greater(rand, 0.9), self.domain2, self.domain1)
        # self.domain = tf.concat([self.domain_0, self.domain_1], 0)
        self.domain = tf.one_hot(self.domain, depth=2)
        self.keep_prob = tf.constant(0.5, dtype=tf.float32)
        # X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        # CNN model for feature extractio
        with tf.name_scope('feature_extractor'):
            self.W_conv0 = weight_variable([256, 1, 1, 64], 'W_Conv0')
            self.b_conv0 = bias_variable([64], 'b_Conv0')
	    # h_conv0 = conv2d(self.X, W_conv0, [1, 31,1,1])+ b_conv0
	    # h_conv0 = tf.contrib.layers.batch_norm(h_conv0, center=True, scale=True, is_training=self.train)
            # h_conv0 = tf.nn.relu(h_conv0)
            # print(self.X.get_shape().as_list())
            h_conv0 = tf.nn.relu(conv2d(self.X, self.W_conv0, [1,31,1,1]) + self.b_conv0)
            tf.Print(h_conv0, [h_conv0])
            h_pool0 = max_pool_2x2(h_conv0, 2,2)
            tf.summary.histogram('Conv 0 weight', self.W_conv0)
            tf.summary.histogram('Conv 0 bias', self.b_conv0)
            tf.summary.histogram('Conv 0 activation', h_conv0)

            self.W_conv1 = weight_variable([15, 1, 64, 128], 'W_conv1')
            self.b_conv1 = bias_variable([128], 'b_conv1')
            # h_conv1 = conv2d(h_pool0, W_conv1, [1,1,1,1])+b_conv1
	    # h_conv1 = tf.contrib.layers.batch_norm(h_conv1, center=True, scale=True, is_training=self.train)
	    # h_conv1 = tf.nn.relu(h_conv1) 
            h_conv1 = tf.nn.tanh(conv2d(h_pool0, self.W_conv1, [1,1,1,1]) + self.b_conv1)
            h_pool1 = max_pool_2x2(h_conv1, 2,2)
            tf.summary.histogram('Conv 1 weight', self.W_conv1)
            tf.summary.histogram('Conv 1 bias', self.b_conv1)
            tf.summary.histogram('Conv 1 activation', h_conv1)
            a, b, c, d = h_pool1.get_shape().as_list()

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, b*c*d])

        # MLP for class prediction
        with tf.name_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size, -1])
#            classify_feats = tf.cond(self.train, source_features, all_features)
            classify_feats = self.feature
            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size, -1])
#            self.classify_labels = tf.cond(self.train, source_labels, all_labels)
            self.classify_labels = self.y
            self.W_fc0 = weight_variable([b*c*d, 1024*2], 'W_fc0')
            self.b_fc0 = bias_variable([1024*2], 'b_fc0')
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, self.W_fc0) + self.b_fc0)
            tf.summary.histogram('fc0 weight', self.W_fc0)
            tf.summary.histogram('fc0 bias', self.b_fc0)
            tf.summary.histogram('fc0 activation', h_fc0)
            h_fc0 = tf.nn.dropout(h_fc0, self.keep_prob)

            self.W_fc1 = weight_variable([1024*2, 1024*2], 'W_fc1')
            self.b_fc1 = bias_variable([1024*2], 'b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, self.W_fc1) + self.b_fc1)
            tf.summary.histogram('fc1 weight', self.W_fc1)
            tf.summary.histogram('fc1 bias', self.b_fc1)
            tf.summary.histogram('fc1 activation', h_fc1)
            h_fc1 = tf.nn.dropout(h_fc1, self.keep_prob)

            self.W_fc2 = weight_variable([1024*2, 1024*2], 'W_fc2')
            self.b_fc2 = bias_variable([1024*2], 'b_fc2')
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1,self.W_fc2) + self.b_fc2)
            tf.summary.histogram('fc2 weight', self.W_fc2)
            tf.summary.histogram('fc2 bias', self.b_fc2)
            tf.summary.histogram('fc2 activation', h_fc2)
            h_fc2 = tf.nn.dropout(h_fc2, self.keep_prob)

            self.W_fc3 = weight_variable([1024*2, 1024*2], 'W_fc3')
            self.b_fc3 = bias_variable([1024*2], 'b_fc3')
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3)
            tf.summary.histogram('fc3 weight', self.W_fc3)
            tf.summary.histogram('fc3 bias', self.b_fc3)
            tf.summary.histogram('fc3 activation', h_fc3)
            h_fc3 = tf.nn.dropout(h_fc3, self.keep_prob)

            self.W_fc4 = weight_variable([1024*2, 1024*2], 'W_fc4')
            self.b_fc4 = bias_variable([1024*2], 'b_fc4')
            h_fc4 = tf.nn.relu(tf.matmul(h_fc3, self.W_fc4) + self.b_fc4)
            tf.summary.histogram('fc4 weight', self.W_fc4)
            tf.summary.histogram('fc4 bias', self.b_fc4)
            tf.summary.histogram('fc4 activation', h_fc4)
            h_fc4 = tf.nn.dropout(h_fc4, self.keep_prob)

            self.W_fc5 = weight_variable([1024*2, 1024*2], 'W_fc5')
            self.b_fc5 = bias_variable([1024*2], 'b_fc5')
            h_fc5 = tf.nn.relu(tf.matmul(h_fc4, self.W_fc5) + self.b_fc5)
            tf.summary.histogram('fc5 weight', self.W_fc5)
            tf.summary.histogram('fc5 bias', self.b_fc5)
            tf.summary.histogram('fc5 activation', h_fc5)
            h_fc5 = tf.nn.dropout(h_fc5, self.keep_prob)

            self.W_fc6 = weight_variable([1024*2, 1024*2], 'W_fc6')
            self.b_fc6 = bias_variable([1024*2], 'b_fc6')
            h_fc6 = tf.nn.relu(tf.matmul(h_fc5, self.W_fc6) + self.b_fc6)
            tf.summary.histogram('fc6 weight', self.W_fc6)
            tf.summary.histogram('fc6 bias', self.b_fc6)
            tf.summary.histogram('fc6 activation', h_fc6)
            h_fc6 = tf.nn.dropout(h_fc6, self.keep_prob)

            self.W_fc7 = weight_variable([1024*2, 1677], 'W_fc7')
            self.b_fc7 = bias_variable([1677], 'b_fc7')

            logits = tf.matmul(h_fc6, self.W_fc7) + self.b_fc7
            tf.summary.histogram('fc7 weight', self.W_fc7)
            tf.summary.histogram('fc7 bias', self.b_fc7)
            tf.summary.histogram('fc7 activation', logits)
        with tf.name_scope('predicted_loss'):

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.name_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([b*c*d, 1024*2], 'd_W_fc0')
            d_b_fc0 = bias_variable([1024*2], 'd_b_fc0')
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)
            tf.summary.histogram('domain fc0 weights', d_W_fc0)
            tf.summary.histogram('domain fc0 bias', d_b_fc0)
            tf.summary.histogram('domain fc0 activation', d_h_fc0)
            d_h_fc0 = tf.nn.dropout(d_h_fc0, keep_prob=self.keep_prob)

            d_W_fc1 = weight_variable([1024*2, 1024*2], 'd_W_fc1')
            d_b_fc1 = bias_variable([1024*2], 'd_b_fc1')
            d_h_fc1 = tf.nn.relu(tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1)
            tf.summary.histogram('domain fc1 weights', d_W_fc1)
            tf.summary.histogram('domain fc1 bias', d_b_fc1)
            tf.summary.histogram('domain fc1 activation', d_h_fc1)
            d_h_fc1 = tf.nn.dropout(d_h_fc1, keep_prob=self.keep_prob)

            d_W_fc2 = weight_variable([1024*2, 1024*2], 'd_W_fc2')
            d_b_fc2 = bias_variable([1024*2], 'd_b_fc2')
            d_h_fc2 = tf.nn.relu(tf.matmul(d_h_fc1, d_W_fc2) + d_b_fc2)
            tf.summary.histogram('domain fc2 weights', d_W_fc2)
            tf.summary.histogram('domain fc2 bias', d_b_fc2)
            tf.summary.histogram('domain fc2 activation', d_h_fc2)
            d_h_fc2 = tf.nn.dropout(d_h_fc2, keep_prob=self.keep_prob)

            d_W_fc3 = weight_variable([1024*2, 1024*2], 'd_W_fc3')
            d_b_fc3 = bias_variable([1024*2], 'd_b_fc3')
            d_h_fc3 = tf.nn.relu(tf.matmul(d_h_fc2, d_W_fc3) + d_b_fc3)
            tf.summary.histogram('domain fc3 weights', d_W_fc3)
            tf.summary.histogram('domain fc3 bias', d_b_fc3)
            tf.summary.histogram('domain fc3 activation', d_h_fc3)
            d_h_fc3 = tf.nn.dropout(d_h_fc3, keep_prob=self.keep_prob)

            d_W_fc4 = weight_variable([1024*2, 1024*2], 'd_W_fc4')
            d_b_fc4 = bias_variable([1024*2], 'd_b_fc4')
            d_h_fc4 = tf.nn.relu(tf.matmul(d_h_fc3, d_W_fc4) + d_b_fc4)
            tf.summary.histogram('domain fc4 weights', d_W_fc4)
            tf.summary.histogram('domain fc4 bias', d_b_fc4)
            tf.summary.histogram('domain fc4 activation', d_h_fc4)
            d_h_fc4 = tf.nn.dropout(d_h_fc4, keep_prob=self.keep_prob)

            d_W_fc5 = weight_variable([1024*2, 2], 'd_W_fc5')
            d_b_fc5 = bias_variable([2], 'd_b_fc5')
            d_logits = tf.matmul(d_h_fc4, d_W_fc5) + d_b_fc5
            tf.summary.histogram('domain fc5 weigth', d_W_fc5)
            tf.summary.histogram('domain fc5 bias', d_b_fc5)
            tf.summary.histogram('domain fc5 activation', d_logits)
        with tf.name_scope('domain_loss'):

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)


graph = tf.get_default_graph()
with graph.as_default():
    model = Acoustic_Model()

    # learning_rate = tf.placeholder(tf.float32, [])
    learning_rate = model.lr
    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss
    with tf.name_scope('Training'):
        regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)


    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    tf.summary.scalar('prediction loss', pred_loss)
    tf.summary.scalar('domain loss', domain_loss)
    tf.summary.scalar('total loss', total_loss)
    tf.summary.scalar('label accuracy', label_acc)
    tf.summary.scalar('domain accuracy', domain_acc)
    summary_op = tf.summary.merge_all()



with tf.Session(graph=graph, config=tf.ConfigProto(intra_op_parallelism_threads=12, allow_soft_placement=True)) as sess:
    print('entered')
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print('initilized')
    coord = tf.train.Coordinator()
    print('initilized2')
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    print('startting sess')
    saver = tf.train.Saver()
    
    sess.graph.finalize()
    k = 0
    avg_loss = []
    avg_accuracy = []
    avg_domain_loss = []
    avg_domain_accuracy = []
    for i in range(num_steps):
        _, ploss, dloss, l_acc, d_acc,lam = sess.run([dann_train_op, pred_loss, domain_loss, label_acc, domain_acc, model.l])
        avg_loss.append(ploss)
        avg_accuracy.append(l_acc)
        avg_domain_loss.append(dloss)
        avg_domain_accuracy.append(d_acc)
        if i%10000 == 0:
            print('epoch:%d/%d  ploss:%f dloss:%f accuracy:%f avg_loss:%f avg_accuracy:%f avg_domain_loss:%f avg_domain_accuracy:%f l_parameter:%f'%(k,i%234375, ploss, dloss,l_acc,sum(avg_loss)/len(avg_loss), sum(avg_accuracy)/len(avg_accuracy), sum(avg_domain_loss)/len(avg_domain_loss), sum(avg_domain_accuracy)/len(avg_domain_accuracy), lam))
        if i%234375 == 0:
            k = k+1
            avg_loss = []
            avg_accuracy = []

        if i%10000 == 0:
            save_path = saver.save(sess, 'saved_model_baseline_adver/baseline_model.ckpt')
    coord.request_stop()
    coord.join(threads)
