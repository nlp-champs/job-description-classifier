#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import re
import json

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./prod", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Passed input from frontend
tf.flags.DEFINE_string("user_input", "", "Passed input from express server from the front end")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x_raw = FLAGS.user_input.split(',')
# x_raw = ["a 401k is included after time", "great knowledge in python is required", "3 years expereince with being a boss", "you must know R programming", "we pride ourselves on our cultural hiring with bachelor's degrees and at least 5 years of experience with benefits", "experience in microsoft word is a plus", "being a noob is preferred", "we want clarkson students with mechanical engineering degrees", "we want greg scholz with 5 years experience electrician"]
x_raw = [clean_str(line) for line in x_raw]
y_test = None # we want to predict

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # scored predictions
        # scored_predictions = graph.get_operation_by_name("output/scored_predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# modifications to the human readable set - the actual names of the classifications
predictions_dict = {}
for prediction_set in predictions_human_readable:
    if prediction_set[1] == '0.0':
        predictions_dict[prediction_set[0]] = "required skill"
    if prediction_set[1] == '1.0':
        predictions_dict[prediction_set[0]] = "degree"
    if prediction_set[1] == '2.0':
        predictions_dict[prediction_set[0]] = "years"
    if prediction_set[1] == '3.0':
        predictions_dict[prediction_set[0]] = "desired skill"
    if prediction_set[1] == '4.0':
        predictions_dict[prediction_set[0]] = "benefits"
    if prediction_set[1] == '5.0':
        predictions_dict[prediction_set[0]] = "culture"
    if prediction_set[1] == '6.0':
        predictions_dict[prediction_set[0]] = "other / no category found"

# return json object to stdout
print json.dumps(predictions_dict)
