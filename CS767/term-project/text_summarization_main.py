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
import time 
from tensorflow.layers import Dense
import process_data as dh 

#model inputs for the model

def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')
    summary_sequence_length = tf.placeholder(tf.int32,(None,), name='summary_sequence_length')
    max_summary_length = tf.reduce_max(summary_sequence_length)
    text_sequence_length = tf.placeholder(tf.int32, (None,), name='text_sequence_length')

    return input_data, targets, learning_rate, keep_probability, summary_sequence_length, max_summary_length, text_sequence_length

#Add <GO> token to tell decoder when to go 
def process_decoding_input(targets, word_to_int, batch_size):
    ending = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    decoder_input = tf.concat([tf.fill([batch_size,1], word_to_int['<GO>']),ending],1)

    return decoder_input

#Define the encoding layer
def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):

    for  layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1, seed = 2))
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob)
            bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1, seed = 2))
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob)

            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,rnn_inputs, sequence_length,dtype=tf.float32)
            encoder_output = tf.concat(encoder_output,2)

            return encoder_output, encoder_state
#Define training step for decoding layer
def decoding_layer_train(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, vocab_size, max_summary_length, keep_prob):

    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=summary_length, time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, initial_state, output_layer)

    training_logits, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_summary_length)

    return training_logits
#Define inference step for decoder
def decoding_layer_infer(embeddings, start_id, end_id, dec_cell, encoder_state, output_layer, max_target_sequence_length, batch_size, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.fill([batch_size], start_id),end_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True, maximum_iterations=max_target_sequence_length)

    return outputs

#Define decoding layer. 

def decoding_layer(dec_input, embeddings, encoder_output, encoder_state, vocab_size, text_length, summary_length, max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):

    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            
    output_layer = Dense(vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))        
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_output, text_length, normalize=False, name='BahdanauAttention')
    lstm_attention = tf.contrib.seq2seq.AttentionWrapper(lstm, attention_mechanism, rnn_size)
    initial_state = lstm_attention.zero_state(batch_size, tf.float32)
    with tf.variable_scope('decode'):
        training_logits = decoding_layer_train(dec_input, summary_length, lstm_attention, initial_state, output_layer, vocab_size, max_summary_length, keep_prob)
    
    with tf.variable_scope('decode', reuse=True):
        inference_logits = decoding_layer_infer(embeddings, vocab_to_int['<GO>'], vocab_to_int['<EOS>'], lstm_attention, initial_state, output_layer, max_summary_length, batch_size, keep_prob)

    return training_logits, inference_logits    


def model(input_data, target_data, embeddings, keep_prob, text_length, summary_length, max_summary_length, vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    encoder_output, encoder_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_decoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    training_logits, inference_logits = decoding_layer(dec_embed_input, embeddings, encoder_output, encoder_state, vocab_size, text_length, summary_length,max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers)

    return training_logits, inference_logits


#get batches

def get_batches(summaries, texts, batch_size, word_to_int):

    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i*batch_size
        summaries_batch = summaries[start_i:start_i+batch_size]
        texts_batch = texts[start_i:start_i+batch_size]
        pad_summaries_batch = np.array(dh.pad_sentence_batch(summaries_batch, word_to_int))
        pad_texts_batch = np.array(dh.pad_sentence_batch(texts_batch, word_to_int))

        #Get batch lengths
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []

        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths       

#Build the graph

def build_graph(vocab_to_int,embeddings, rnn_size, num_layers, batch_size):
    print('Building graph...')
    train_graph = tf.Graph()

    with train_graph.as_default():
        input_data, targets, learning_rate, keep_prob, summary_length, max_sentence_length, text_length = model_inputs()

        #Creating training and inference logits
        training_logits, inference_logits = model(tf.reverse(input_data, [-1]), targets,embeddings, keep_prob,text_length, summary_length, max_sentence_length, len(vocab_to_int)+1, rnn_size, num_layers, vocab_to_int, batch_size)
        training_logits = tf.identity(training_logits.rnn_output, name='logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

        #weights for the sequence_loss
        masks = tf.sequence_mask(summary_length, max_sentence_length, dtype=tf.float32, name='masks')

        #Optimizer
        with tf.name_scope("optimization"):
            #loss function 
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
            
            #Optimizer 
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
            #Gradient clipping
            gradients = optimizer.compute_gradients(cost)

            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]

            train_op = optimizer.apply_gradients(capped_gradients)

            print('Building graph finished.')

    return train_graph, cost, train_op, input_data, targets, learning_rate, keep_prob, summary_length, max_sentence_length, text_length

def main():
    #assume we take arguments
    #TODO: Add theses as user inputs
    learning_rate_dacay = 0.95 
    min_learning_rate = 0.0005 
    display_step = 20
    stop_early = 0
    stop = 3
    per_epoch = 3
    rnn_size = 256
    num_layers = 2
    batch_size = 1 
    epochs = 80
    keep_probability = 0.75
    update_loss = 0
    batch_loss = 0
    summary_update_loss = []
    learning_rate = 0.005


    #get text
    stories, summaries, word_count = dh.load_stories('../data/cnn/stories', num=500)
    embeddings,word_to_int, int_to_word=dh.build_vocab(word_count)
    clean_stories =dh.convert_text_to_int(stories, word_to_int, eos=True)
    clean_summaries =dh.convert_text_to_int(summaries, word_to_int)
    
    update_check = (len(clean_stories)//batch_size//per_epoch)-1 
    #Build graph
    train_graph, cost_func, train_op, input_data, targets, lr, keep_prob, summary_length, max_sentence_length, text_length = build_graph(word_to_int, embeddings, rnn_size, num_layers, batch_size)

    cwd = os.getcwd()
    checkpoint =os.path.join(cwd,"best_model.ckpt")
    

    with tf.Session(graph = train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epochs + 1):
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(get_batches(clean_summaries, clean_stories,batch_size, word_to_int)):
                start_time = time.time()
                _, loss = sess.run([train_op, cost_func], 
                        {input_data:texts_batch, 
                            targets:summaries_batch, 
                            lr:learning_rate,
                            summary_length:summaries_lengths,
                            text_length:texts_lengths,
                            keep_prob:keep_probability})
        
                batch_loss  += loss 
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i & display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f})'.format(epoch_i, epochs, batch_i, len(clean_stories)//batch_size, batch_loss/display_step, batch_time*display_step))
                    batch_loss = 0
                
                if batch_i & update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss/update_check,3))
                    summary_update_loss.append(update_loss)


                    #Save minimums
                    if update_loss <= min(summary_update_loss):
                        print("New Record!")
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)
                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break 
                    
                    update_loss = 0
    
                    # Reduce learning rate, but not below its minimum value
            learning_rate *= 0.95 
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
        
            if stop_early == stop:
                print("Stopping Training.")
                break 

if __name__ == "__main__":
    main()
    

        
