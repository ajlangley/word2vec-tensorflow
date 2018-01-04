import glob
import nltk
import numpy as np
import os
import string
import pandas


class TextStream:
    def __init__(self, input_dir):
        self.infile = input_dir
        self.generator = raw_corpus_generator

    def __iter__(self):
        return self.generator(self.infile)


class NGramStream:
    def __init__(self, infile):
        self.infile = infile
        self.generator = ngram_generator

    def __iter__(self):
        return self.generator(self.infile)


def raw_corpus_generator(input_dir):
    for filename in glob.glob(os.path.join('./data/{}/'.format(input_dir), '*.csv')):
        sentences = pandas.read_csv(filename,
                                    delimiter='\n',
                                    header=None,
                                    error_bad_lines=False,
                                    warn_bad_lines=False)
        for sentence in sentences.itertuples():
            for word in nltk.word_tokenize(sentence[1].lower()):
                if word not in string.punctuation:
                    yield word


def ngram_generator(input_dir):
    infile = './training-sets/{}/train.csv'.format(input_dir, input_dir)

    ngrams = pandas.read_csv(infile,
                             delimiter='\n',
                             header=None,
                             error_bad_lines=False,
                             warn_bad_lines=False)

    for row in ngrams.itertuples():
        training_example = np.int32(np.asarray(row[1].split()))
        input_token = training_example[0]
        context_tokens = training_example[1:]

        if len(context_tokens):
            yield input_token, context_tokens
