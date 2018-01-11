import tensorflow as tf
from tensorflow.python import debug as tf_debug
import math
import os
from utils import get_negative_sample


class Word2Vec:
    def __init__(self, vocabulary_size, embedding_size):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

    def __build_training_graph__(self, learning_rate, negative_sample_size, debug):
        self.training_graph = tf.Graph()
        self.training_session = tf.Session(graph=self.training_graph)
        if debug:
            self.training_session = tf_debug.LocalCLIDebugWrapperSession(self.training_session)
            self.training_session.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        with self.training_graph.as_default():
            self.W_in = tf.get_variable(name='W_in',
                                        trainable=True,
                                        initializer=tf.random_uniform((self.vocabulary_size,
                                                                       self.embedding_size),
                                                                      -1.0, 1.0))
            self.W_out = tf.get_variable(name='W_out',
                                         trainable=True,
                                         initializer=tf.truncated_normal((self.vocabulary_size,
                                                                          self.embedding_size),
                                                                         stddev=1.0 / math.sqrt(self.embedding_size)))

            self.input_label = tf.placeholder(tf.int32,
                                              shape=(),
                                              name='input_label')
            self.context_labels = tf.placeholder(tf.int32,
                                                 shape=(None,),
                                                 name='context_label')
            negative_sample = tf.py_func(func=get_negative_sample,
                                         inp=[self.input_label,
                                              self.context_labels,
                                              self.vocabulary_size,
                                              negative_sample_size],
                                         Tout=tf.int32)

            v_input = tf.nn.embedding_lookup(
                params=self.W_in,
                ids=self.input_label,
            )
            v_context = tf.nn.embedding_lookup(
                params=self.W_out,
                ids=self.context_labels
            )
            v_neg = tf.nn.embedding_lookup(
                params=self.W_out,
                ids=negative_sample
            )

            pos_logits = tf.reduce_sum(tf.log(tf.clip_by_value(tf.sigmoid(tf.tensordot(v_context, v_input,
                                                                                        axes=[[1], [0]])), 1e-8, 1.0)))
            neg_logits = tf.reduce_sum(tf.log(tf.clip_by_value(tf.sigmoid(tf.tensordot(-v_neg, v_input,
                                                                                       axes=[[1], [0]])), 1e-8, 1.0)))

            self.loss = -pos_logits - neg_logits
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

            init_op = tf.global_variables_initializer()
            self.training_session.run(init_op)

            self.saver = tf.train.Saver([self.W_in])

        self.training_graph.finalize()

    def __build_validation_graph__(self, n_validation_tokens, n_most_similar):
        self.validation_graph = tf.Graph()
        self.validation_session = tf.Session(graph=self.validation_graph)

        with self.validation_graph.as_default():
            self.validation_tokens = tf.placeholder(dtype=tf.int32,
                                                    name='validation_tokens',
                                                    shape=(n_validation_tokens,), )
            self.embeddings = tf.placeholder(dtype=tf.float32,
                                             name='validation_embeddings',
                                             shape=(self.vocabulary_size,
                                                    self.embedding_size))

            vocab_norm = self.embeddings / tf.reshape(tf.norm(self.embeddings,
                                                              ord=2,
                                                              axis=1),
                                                      (self.vocabulary_size, 1))

            v_validation = tf.nn.embedding_lookup(
                params=vocab_norm,
                ids=self.validation_tokens
            )

            cosine_distance = tf.matmul(v_validation,
                                        vocab_norm,
                                        transpose_b=True)
            _, self.most_similar = tf.nn.top_k(cosine_distance,
                                               k=n_most_similar + 1,
                                               sorted=True)

        self.validation_graph.finalize()

    def sgd_step(self, input_token, context_tokens):
        feed_dict = {self.input_label: input_token,
                     self.context_labels: context_tokens}
        _, loss = self.training_session.run([self.optimizer, self.loss], feed_dict=feed_dict)

        return loss

    def validate(self, validation_tokens):
        print('\n\tPerforming validation...')
        feed_dict = {self.validation_tokens: validation_tokens,
                     self.embeddings: self.W_in.eval(self.training_session)}
        most_similar = self.validation_session.run(self.most_similar, feed_dict=feed_dict)

        return most_similar[:, 1:]

    def save(self, log_dir, step):
        filepath = os.path.join('training-logs', log_dir)
        if not os.path.exists(filepath):
            os.mkdir('training-logs')
            
        self.saver.save(self.training_session, os.path.join(filepath, 'embeddings.ckpt'), step)
