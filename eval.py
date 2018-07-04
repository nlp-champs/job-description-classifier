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

# Data Parameters
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("neutral_data_file", "./data/rt-polaritydata/rt-polarity.neu", "Data source for the neutral data.")

tf.flags.DEFINE_string("tasks_data_file", "./data/monster-jobs/tasks_new.txt", "Data source for the tasks snippets.")
tf.flags.DEFINE_string("culture_data_file", "./data/monster-jobs/culture_new.txt", "Data source for the culture snippets.")
tf.flags.DEFINE_string("required_data_file", "./data/monster-jobs/required_new.txt", "Data source for the required skill snippets.")
tf.flags.DEFINE_string("degree_data_file", "./data/monster-jobs/degree_new.txt", "Data source for the degree snippets.")
tf.flags.DEFINE_string("desired_data_file", "./data/monster-jobs/desired_new.txt", "Data source for the desired skill snippets.")
tf.flags.DEFINE_string("years_data_file", "./data/monster-jobs/years_new.txt", "Data source for the years snippets.")
tf.flags.DEFINE_string("benefits_data_file", "./data/monster-jobs/benefits_new.txt", "Data source for the years snippets.")
tf.flags.DEFINE_string("other_data_file", "./data/monster-jobs/other_new.txt", "Data source for the 'other' snippets.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels([FLAGS.required_data_file, FLAGS.degree_data_file, FLAGS.years_data_file, FLAGS.desired_data_file, FLAGS.benefits_data_file, FLAGS.culture_data_file, FLAGS.other_data_file])
    y_test = np.argmax(y_test, axis=1)
else:
    # x_raw = ["required knowledge includes knowing MATLAB", "degree in electrical engineering"]
    # y_test = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]
    y_test = None
    #sRawText = list(open("./data/monster-jobs/job_lines_raw.txt", "r").readlines())
    #sRawText = [s.strip() for s in sRawText]
    x_raw = ["a 401k is included after time", "great knowledge in python is required", "3 years expereince with being a boss", "you must know R programming", "we pride ourselves on our cultural hiring with bachelor's degrees and at least 5 years of experience with benefits", "experience in microsoft word is a plus", "being a noob is preferred", "we want clarkson students with mechanical engineering degrees", "we want greg scholz with 5 years experience electrician"]
    #x_raw = [clean_str(line) for line in sRawText]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

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
        # scored_predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# modifications to the human readable set - the actual names of the classifications
predictions_human_readable_final = [['prediction', 'classification']] # header for the 'final' csv
for prediction_set in predictions_human_readable:
    if prediction_set[1] == '0.0':
        prediction_set[1] = "required skill"
    if prediction_set[1] == '1.0':
        prediction_set[1] = "degree"
    if prediction_set[1] == '2.0':
        prediction_set[1] = "years"
    if prediction_set[1] == '3.0':
        prediction_set[1] = "desired skill"
    if prediction_set[1] == '4.0':
        prediction_set[1] = "benefits"
    if prediction_set[1] == '5.0':
        prediction_set[1] = "culture"
    if prediction_set[1] == '6.0':
        prediction_set[1] = "other / no category found"
    predictions_human_readable_final.append(prediction_set)

out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable_final)
