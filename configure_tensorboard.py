import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log-dir', type=str, dest='log_dir', default='log_dir')
args = parser.parse_args()
log_dir = args.log_dir

log_path = os.path.join('training-logs', log_dir)
if not os.path.exists(log_path):
    print('[ERROR]: The training log directory you entered does not exist. Exiting.')
    exit()

summary_writer = tf.summary.FileWriter(log_path)
config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = 'W_in'
embedding.metadata_path = 'vocab.tsv'

projector.visualize_embeddings(summary_writer, config)
