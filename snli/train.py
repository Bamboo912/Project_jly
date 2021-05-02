# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import json
import datetime
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf
from clstm_classifier import clstm_clf

###### 这段作用是啥
try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    error = "Please install scikit-learn."
    print(str(e) + ': ' + error)
    sys.exit()

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Model choices
tf.flags.DEFINE_string('clf', 'clstm', "Type of classifiers. Default: cnn. You have four choices: [cnn, lstm, blstm, clstm]")

# Data parameters
tf.flags.DEFINE_string('data_file', 'E:\\Lessons\\MachineLearning\\Project\\TextClassification-master\\data\\data.csv', 'Data file path')
tf.flags.DEFINE_string('stop_word_file', None, 'Stop word file path')
tf.flags.DEFINE_string('language', 'en', "Language of the data file. You have two choices: [ch, en]")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 0, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0.1, 'Cross validation test size')

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 256, 'Word embedding size. For CNN, C-LSTM.')
tf.flags.DEFINE_string('filter_sizes', '3, 4, 5', 'CNN filter sizes. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the LSTM cell. For LSTM, Bi-LSTM')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')  # All
tf.flags.DEFINE_float('l2_reg_lambda', 0.001, 'L2 regularization lambda')  # All

# Training parameters
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate. Range: (0, 1]')  # Learning rate decay
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')  # Learning rate decay
tf.flags.DEFINE_integer('evaluate_every_steps', 100, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')



FLAGS = tf.app.flags.FLAGS    #### 奥，直接用FLAGS 代替tf.app.flags，封装好的结构体
tf.app.flags.DEFINE_string('f', '', 'kernel') ##add by me 不懂加这行的原因

if FLAGS.clf == 'lstm':
    FLAGS.embedding_size = FLAGS.hidden_size
elif FLAGS.clf == 'clstm':
    FLAGS.hidden_size = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load and save data
# =============================================================================
###### 仔细看一下data_helper 文件
data, labels, lengths, vocab_processor = data_helper.load_data(file_path=FLAGS.data_file,
                                                               sw_path=FLAGS.stop_word_file,
                                                               min_frequency=FLAGS.min_frequency,
                                                               max_length=FLAGS.max_length,
                                                               language=FLAGS.language,
                                                               shuffle=True)

# Save vocabulary processor   这些参数vocab_size，max_length也和data_helper有关
vocab_processor.save(os.path.join(outdir, 'vocab'))

FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)

FLAGS.max_length = vocab_processor.max_document_length

params = FLAGS.flag_values_dict()   #####这个命令把前面FLAGS定义的参数归到了paras中
# Print parameters
model = params['clf']
if model == 'cnn':
    del params['hidden_size']
    del params['num_layers']
elif model == 'lstm' or model == 'blstm':
    del params['num_filters']
    del params['filter_sizes']
    params['embedding_size'] = params['hidden_size']
elif model == 'clstm':
    params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']

params_dict = sorted(params.items(), key=lambda x: x[0])
print('Parameters:')
for item in params_dict:
    print('{}: {}'.format(item[0], item[1]))
print('')

# Save parameters to file
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file, True)
params_file.close()


# Simple Cross validation   train_test_split是库里面的函数，data是前面用data_helper.load data加载进来的,用来划分训练集和测试集的
x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(data,
                                                                                    labels,
                                                                                    lengths,
                                                                                    test_size=FLAGS.test_size,
                                                                                    random_state=22)
# Batch iterator
train_data = data_helper.batch_iter(x_train, y_train, train_lengths, FLAGS.batch_size, FLAGS.num_epochs)
#####查一下这个是干啥的:训练数据
# Train
# =============================================================================

with tf.Graph().as_default():
    with tf.Session() as sess:
        if FLAGS.clf == 'cnn':
            classifier = cnn_clf(FLAGS)   ####不用管数据的吗，只输入参数的吗
        elif FLAGS.clf == 'lstm' or FLAGS.clf == 'blstm':
            classifier = rnn_clf(FLAGS)
        elif FLAGS.clf == 'clstm':
            classifier = clstm_clf(FLAGS)
        else:
            raise ValueError('clf should be one of [cnn, lstm, blstm, clstm]')  ###确定模型

        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)   ###定义给学习率递减用
        # Learning rate decay
        starter_learning_rate = FLAGS.learning_rate   ###定义给学习率递减用
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)  ###学习率递减
        optimizer = tf.train.AdamOptimizer(learning_rate) ###选择参数优化方法
        grads_and_vars = optimizer.compute_gradients(classifier.cost)  ###计算cost函数的梯度
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) ###结合优化方法和梯度

        # Summaries   summary这个干啥？
        loss_summary = tf.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)

        # Train summary  写入了文件
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summary 写入了文件
        valid_summary_op = tf.summary.merge_all()
        valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # 后面用了saver一次，但是这是干啥的不清楚
        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)

        # 这是干啥的不清楚
        sess.run(tf.global_variables_initializer())


        # 定义训练函数
        def run_step(input_data, is_training=True):
            """Run one step of the training process."""
            input_x, input_y, sequence_length = input_data



            # 参数已在cnn，clstm中训练好
            fetches = {'step': global_step,
                       'cost': classifier.cost,
                       'accuracy': classifier.accuracy,
                       'learning_rate': learning_rate}

            # cnn，clstm模型用的input_x
            feed_dict = {classifier.input_x: input_x,
                         classifier.input_y: input_y}

            # fetches, feed_dict是什么封装的结构，干什么的，这里咋这样训练，咋还用到了accuracy和summaries
            if FLAGS.clf != 'cnn':
                fetches['final_state'] = classifier.final_state
                feed_dict[classifier.batch_size] = len(input_x)
                feed_dict[classifier.sequence_length] = sequence_length

            if is_training:
                fetches['train_op'] = train_op
                fetches['summaries'] = train_summary_op
                feed_dict[classifier.keep_prob] = FLAGS.keep_prob
            else:
                fetches['summaries'] = valid_summary_op
                feed_dict[classifier.keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            accuracy = vars['accuracy']
            summaries = vars['summaries']

            # Write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                valid_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))

            return accuracy


        print('Start training ...')

        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)  #这是个提取当前步骤的命令

            ## 多少步训练和交叉验证
            if current_step % FLAGS.evaluate_every_steps == 0:
                print('\nValidation')
                run_step((x_valid, y_valid, valid_lengths), is_training=False)
                print('')
            ## 多少步保存
            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)

        print('\nAll the files have been saved to {}\n'.format(outdir))