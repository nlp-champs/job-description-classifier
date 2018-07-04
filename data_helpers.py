import numpy as np
import re
import itertools
from collections import Counter


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


def load_data_and_labels(lDataFilePaths):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # neutral_examples = list(open(neutral_data_file, "r").readlines())
    # neutral_examples = [s.strip() for s in neutral_examples]
    lExamples = []
    for sFilePath in lDataFilePaths:
        sExamples = list(open(sFilePath, "r").readlines())
        sExamples = [s.strip() for s in sExamples]
        lExamples.append(sExamples)

    # Split by words
    # x_text = positive_examples + negative_examples + neutral_examples
    x_text = lExamples[0] + lExamples[1] + lExamples[2] + lExamples[3] + lExamples[4] + lExamples[5] + lExamples[6]
    # x_test =
    # x_test = lExamples[0:6] #
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    # positive_labels = [[0, 1, 0] for _ in positive_examples]
    # negative_labels = [[1, 0, 0] for _ in negative_examples]
    # neutral_labels = [[0, 0, 1] for _ in neutral_examples]

    # labels for job stuff, ordered by our defined 'priority' TODO: make this work for any number of catageories
    required_labels = [[1, 0, 0, 0, 0, 0, 0] for _ in lExamples[0]]
    degree_labels   = [[0, 1, 0, 0, 0, 0, 0] for _ in lExamples[1]]
    years_labels    = [[0, 0, 1, 0, 0, 0, 0] for _ in lExamples[2]]
    desired_labels  = [[0, 0, 0, 1, 0, 0, 0] for _ in lExamples[3]]
    benefits_labels = [[0, 0, 0, 0, 1, 0, 0] for _ in lExamples[4]]
    culture_labels  = [[0, 0, 0, 0, 0, 1, 0] for _ in lExamples[5]]
    other_labels    = [[0, 0, 0, 0, 0, 0, 1] for _ in lExamples[6]]

    # y = np.concatenate([positive_labels, negative_labels, neutral_labels], 0)
    y = np.concatenate([required_labels, degree_labels, years_labels, desired_labels, benefits_labels, culture_labels, other_labels], 0)
    print str(y)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
