import argparse
import datetime
import os
import shutil
from read_data import NGramStream
from utils import load_vocab, display_elapsed_time
from word2vec import Word2Vec

UNKNOWN_TOKEN = '<UNK>'

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--learning-rate', type=float, dest='learning_rate', default=.025)
parser.add_argument('-n', '--num-epochs', type=int, dest='n_epochs', default=5)
parser.add_argument('-e', '--embedding-size', type=int, dest='embedding_size', default=128)
parser.add_argument('-k', '--negative-sample-size', type=int, dest='negative_sample_size', default=8)
parser.add_argument('-i', '--input-dir', type=str, dest='input_dir')
parser.add_argument('-l', '--log-dir', type=str, dest='log_dir', default='trained-embeddings')
parser.add_argument('-v', '--validate-every', type=int, dest='validate_every', default=1000000)
parser.add_argument('-d', '--debug', dest='debug', action='store_true')
args = parser.parse_args()
learning_rate = args.learning_rate
nPasses = args.n_epochs
embedding_size = args.embedding_size
negative_sample_size = args.negative_sample_size
input_dir = args.input_dir
log_dir = args.log_dir
validate_every = args.validate_every
debug = args.debug

index_to_word, word_to_index = load_vocab(input_dir)
vocabulary_size = len(index_to_word)
ngrams = NGramStream(input_dir)

model = Word2Vec(vocabulary_size, embedding_size)
model.__build_training_graph__(learning_rate=learning_rate,
                               negative_sample_size=negative_sample_size,
                               debug=debug)
model.__build_validation_graph__(n_validation_tokens=4, n_most_similar=5)

validation_tokens = [word_to_index['he'], word_to_index['i'], word_to_index['this'], word_to_index['you']]

start_time = datetime.datetime.now()
total_steps = 0
loss_since_update = 0
steps_since_update = 0

log_path = os.path.join('training-logs', log_dir)
if os.path.exists(log_path):
    selection = input('The directory {} already exists. Would you like to overwrite it? [Y/N]: '.format(log_path))
    while selection is not 'Y' and selection is not 'y' and selection is not 'N' and selection is not 'n':
        selection = input('You did not enter a valid option. Choose from [Y/N]: ')
    if selection is 'Y' or selection is 'y':
        shutil.rmtree(log_path)
        print('\n')
    elif selection is 'N' or selection is 'n':
        print('Exiting.')
        exit()


print('Training Word2Vec model with vocabulary size {} and embedding size {}...\n'.format(vocabulary_size,
                                                                                          embedding_size))
for epoch in range(nPasses):
    print('\nBeginning training epoch {}...'.format(epoch + 1))

    for input_token, context in ngrams.__iter__():
        loss = model.sgd_step(input_token, context)
        loss_since_update += loss
        steps_since_update += 1
        total_steps += 1

        if not steps_since_update % validate_every:
            if not debug:
                most_similar_tokens = model.validate(validation_tokens)
                for i, validation_token in enumerate(validation_tokens):
                    most_similar_words = [index_to_word[word_token] for word_token in most_similar_tokens[i]]

                    print('\tWords most similar to {}: {}'.format(index_to_word[validation_token],
                                                                  ', '.join(most_similar_words)))
                print('\t[AVG ERROR FOR LAST {} STEPS]: {}'.format(validate_every,
                                                                   loss_since_update / steps_since_update))
                display_elapsed_time(start_time)

                model.save(log_dir, total_steps)

            loss_since_update = 0
            steps_since_update = 0

    print('Training epoch {} complete.'.format(epoch + 1))
    display_elapsed_time(start_time)

    total_loss = 0
    since_update = 0
