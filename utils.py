import nltk
import numpy as np
import pandas
from datetime import datetime as dt
import glob
import os
import csv
import string
from read_data import TextStream

UNKNOWN_TOKEN = '<UNK>'


def build_vocab(input_dir, output_dir, vocabulary_size):
    print('\nBuilding model vocabulary with {} most frequent words...'.format(vocabulary_size))

    sentences = TextStream(input_dir)
    word_frequencies = nltk.FreqDist(sentences.__iter__())
    vocab_frequencies = word_frequencies.most_common(vocabulary_size - 1)

    index_to_word = [w[0] for w in vocab_frequencies] + [UNKNOWN_TOKEN]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    filepath = './training-sets/' + output_dir + '/' + 'vocab'

    print('Vocabulary generation complete.\n')

    np.save(filepath, index_to_word)

    return index_to_word, word_to_index


def build_training_set(input_dir, output_dir, index_to_word, word_to_index, context_radius):
    print('Building training set...')

    filepath = './training-sets/' + output_dir + '/train.csv'

    for filename in glob.glob(os.path.join('./data/{}/'.format(input_dir), '*.csv')):
        sentences = pandas.read_csv(filename,
                                    delimiter='\n',
                                    header=None,
                                    error_bad_lines=False,
                                    warn_bad_lines=False)
        for sentence in sentences.itertuples():
            tokenized_sentence = [word for word in nltk.word_tokenize(sentence[1].lower())
                                  if word not in string.punctuation]
            tokenized_sentence = [word_to_index[w] if w in index_to_word else word_to_index[UNKNOWN_TOKEN]
                                  for w in tokenized_sentence]
            for i, input_token in enumerate(tokenized_sentence):
                if input_token is not word_to_index[UNKNOWN_TOKEN]:
                    context = tokenized_sentence[max(0, i - context_radius): i] + \
                              tokenized_sentence[i + 1: min(len(tokenized_sentence), i + 1 + context_radius)]
                    with open(filepath, 'a') as f:
                        writer = csv.writer(f, delimiter=' ')
                        writer.writerow([input_token] + context)

    print('Training set building complete.\n')


def load_vocab(input_dir):
    filepath = './training-sets/' + input_dir + '/vocab.npy'

    index_to_word = np.load(filepath)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    return index_to_word, word_to_index


def get_negative_sample(target, context, vocabulary_size, negative_sample_size):
    negative_sample = []

    while len(negative_sample) < negative_sample_size:
        negative_token = np.random.randint(vocabulary_size)
        if negative_token is not target and negative_token not in context:
            negative_sample.append(negative_token)

    return np.int32(np.asarray(negative_sample))


def display_elapsed_time(start_time):
    time_delta = dt.now() - start_time
    days_elapsed = time_delta.days
    hours_elapsed = int(time_delta.seconds / 3600)
    minutes_elapsed = int(time_delta.seconds / 60) - (hours_elapsed * 60)
    seconds_elapsed = time_delta.seconds - (minutes_elapsed * 60 + hours_elapsed * 60)

    if days_elapsed:
        print('\t[TIME ELAPSED]: {} days, {} minutes, {} seconds'.format(days_elapsed,
                                                                         minutes_elapsed,
                                                                         seconds_elapsed))
    if not days_elapsed and hours_elapsed:
        print('\t[TIME ELAPSED]: {} days, {} hours, {} minutes, {} seconds'.format(days_elapsed,
                                                                                   hours_elapsed,
                                                                                   minutes_elapsed,
                                                                                   seconds_elapsed))
    else:
        print('\t[TIME ELAPSED]: {} minutes, {} seconds'.format(minutes_elapsed, seconds_elapsed))
