import argparse
import os
import shutil
from utils import build_vocab, build_training_set

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vocab-size', type=int, dest='vocabulary_size', default=10000)
parser.add_argument('-i', '--input-dir', type=str, dest='input_dir')
parser.add_argument('-o', '--output-dir', type=str, dest='output_dir')
parser.add_argument('-c', '--context_radius', type=int, dest='context_radius', default=5)
args = parser.parse_args()
vocabulary_size = args.vocabulary_size
input_dir = args.input_dir
output_dir = args.output_dir
context_radius = args.context_radius

filepath = './training-sets/' + output_dir

if not os.path.exists('./training-sets/' + output_dir):
    os.makedirs(filepath)
else:
    selection = input('The directory /training-sets/{}/ already exists. Would you '
                      'like to overwrite it? [Y/N]: '.format(output_dir))
    while selection is not 'Y' and selection is not 'y' and selection is not 'N' and selection is not 'n':
        selection = input('You did not enter a valid option. Choose from [Y/N]: ')
    if selection is 'Y' or selection is 'y':
        shutil.rmtree(filepath)
        os.makedirs(filepath)
    elif selection is 'N' or selection is 'n':
        print('Exiting.')
        exit()

    index_to_word, word_to_index = build_vocab(input_dir=input_dir,
                                               output_dir=output_dir,
                                               vocabulary_size=vocabulary_size)
    build_training_set(input_dir=input_dir,
                       output_dir=output_dir,
                       index_to_word=index_to_word,
                       word_to_index=word_to_index,
                       context_radius=context_radius)
