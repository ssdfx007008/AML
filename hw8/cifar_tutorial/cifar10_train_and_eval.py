# !!! Note for cifar10_train_and_eval.py !!!
#
# 1. Put this file into tensorflow/models/image/cifar10 directory.
# 2. For this file to work, you need to comment out tf.image_summary() in
#      file tensorflow/models/image/cifar_input.py
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'E:/Documents/Git/AML/hw8/logs/cifa_tut',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def evaluate_set (sess, top_k_op, num_examples):
  """Convenience function to run evaluation for for every batch. 
     Sum the number of correct predictions and output one precision value.
  Args:
    sess:          current Session
    top_k_op:      tensor of type tf.nn.in_top_k
    num_examples:  number of examples to evaluate
  """
  num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
  true_count = 0  # Counts the number of correct predictions.
  total_sample_count = num_iter * FLAGS.batch_size

  for step in xrange(num_iter):
    predictions = sess.run([top_k_op])
    true_count += np.sum(predictions)

  # Compute precision
  return true_count / total_sample_count



def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    with tf.variable_scope(tf.get_variable_scope(), reuse=None) as scope:
      global_step = tf.Variable(0, trainable=False)

      # Get images and labels for CIFAR-10.
      images, labels = cifar10.distorted_inputs()      
      images_eval, labels_eval = cifar10.inputs(eval_data=True)

      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = cifar10.inference(images)
      scope.reuse_variables()
      logits_eval = cifar10.inference(images_eval)

      # Calculate loss.
      loss = cifar10.loss(logits, labels)

      # For evaluation
      top_k      = tf.nn.in_top_k (logits,      labels,      1)
      top_k_eval = tf.nn.in_top_k (logits_eval, labels_eval, 1)

      # Add precision summary
      summary_train_prec = tf.placeholder(tf.float32)
      summary_eval_prec  = tf.placeholder(tf.float32)
      tf.summary.scalar('precision/train', summary_train_prec)
      tf.summary.scalar('precision/eval',  summary_eval_prec)

      # Build a Graph that trains the model with one batch of examples and
      # updates the model parameters.
      train_op = cifar10.train(loss, global_step)

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables())

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()

      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()

      # Start running operations on the Graph.
      sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement))
      print('init')
      sess.run(init)
      print('init_complete')

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir,sess.graph)
    
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
            print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            EVAL_STEP = 100
            EVAL_NUM_EXAMPLES = 1024
            if step % EVAL_STEP == 0:
                prec_train = evaluate_set (sess, top_k,      EVAL_NUM_EXAMPLES)
                prec_eval  = evaluate_set (sess, top_k_eval, EVAL_NUM_EXAMPLES)
                print('%s: precision train = %.3f' % (datetime.now(), prec_train))
                print('%s: precision eval  = %.3f' % (datetime.now(), prec_eval))

        if step % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict={summary_train_prec: prec_train,
                                                      summary_eval_prec:  prec_eval})
            summary_writer.add_summary(summary_str, step)

  # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
