#This is the main module
"""
Functions
1.Builds model inputs
2.Encoder
3.Decoder
4.Interpreting results

"""
import tensorflow as tf 
import numpy as np 
import os 
import argparse 

#model inputs for the model

def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')
    summary_sequence_length = tf.placeholder(tf.int32,[None], name='summary_sequence_length')
    max_summary_length = tf.reduce_max(summary_sequence_length)
    text_sequence_length = tf.placeholder(tf.int32, [None], name='text_sequence_length')

    return input_data, targets, learning_rate, keep_probability, summary_sequence_length, max_summary_length, text_sequence_length

#Add <GO> token to tell decoder when to go 
def process_encoding_input(targets, word_to_int, batch_size):
    ending = tf.stride_slice(targets, [0,0], [batch_size, -1], [1,1])
    decoder_input = tf.concat(tf.fill([batch_size,1], word_to_int['<GO>']),ending,1)

    return decoder_input

#Define the encoding layer
def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):

    for  layer in range(num_layers):
        with tf.variables_scope('encoder_{}'.format(layer)):
            fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1, seed = 2))
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob)
            bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1, seed = 2))
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob)

            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,sequence_length,dtype=tf.float32)
            encoder_output = tf.concat(encoder_output,2)

            return encoder_output, encoder_state
