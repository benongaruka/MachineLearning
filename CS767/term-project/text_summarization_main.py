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
import evaluation_helper as evalhelper
import sys, getopt

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
        training_logits, inference_logits = model(tf.reverse(input_data, [-1]), 
                targets,
                embeddings, 
                keep_prob,
                text_length, 
                summary_length, 
                max_sentence_length, 
                len(vocab_to_int)+1, 
                rnn_size, 
                num_layers, 
                vocab_to_int, 
                batch_size)
        training_logits = tf.identity(training_logits.rnn_output, name='logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

        #weights for the sequence_loss
        masks = tf.sequence_mask(summary_length, max_sentence_length, dtype=tf.float32, name='masks')

        #Optimizer
        with tf.name_scope("optimization"):
            #loss function 
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

            tf.summary.scalar('loss',cost)
            
            #Optimizer 
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
            #Gradient clipping
            gradients = optimizer.compute_gradients(cost)

            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]

            train_op = optimizer.apply_gradients(capped_gradients)

            for index, grad in enumerate(capped_gradients):
                tf.summary.histogram("{}-grad".format(capped_gradients[index][1].name), capped_gradients[index])
            
            print('Building graph finished.')
            merge_op = tf.summary.merge_all()

    return train_graph, cost, train_op, input_data, targets, learning_rate, keep_prob, summary_length, max_sentence_length, text_length, merge_op

def main(**kwargs):
    #assume we take arguments
    test_ = kwargs['Test']
    file_loc = kwargs['file_loc']
    num_ = kwargs['n']
    if not test_:
        batch_size = kwargs['batch_s']
        epochs = kwargs['epoch_n']
    learning_rate_dacay = 0.95 
    min_learning_rate = 0.0005 
    display_step = 20
    stop_early = 0
    stop = 100 
    per_epoch = 3
    rnn_size = 256
    num_layers = 2
    keep_probability = 0.75
    update_loss = 0
    batch_loss = 0
    summary_update_loss = []
    learning_rate = 0.005

    #get text
    if test_:
        stories, summaries, word_count = dh.load_stories(file_loc, test = test_, num = num_)
        embeddings,word_to_int, int_to_word=dh.build_vocab(word_count)
        X_test = dh.convert_text_to_int(stories, word_to_int) # we don't need EOS, we will loop through these ourselves
        Y_test = dh.convert_text_to_int(summaries, word_to_int)
        
        batch_size = 30 
        #Create graph
        train_graph, cost_func, train_op, input_data, targets, lr, keep_prob, summary_length, max_sentence_length, text_length,summary_op = build_graph(word_to_int, 
                embeddings, rnn_size, num_layers, batch_size)

        #Test model 
        summary_lengths = []
        for sentence in Y_test:
            summary_lengths.append(len(sentence))

        avg_summary_length = sum(summary_lengths)//len(summary_lengths)
        y_sys = test_model(X_test, Y_test, './best_model.ckpt', train_graph,batch_size, avg_summary_length) # How do you test without knowing training batch_size
        h_summaries = dh.convert_int_to_text(y_sys, int_to_word, word_to_int['<PAD>'])
        rouge_scores = evalhelper.evaluate_rouge_score(h_summaries, summaries)
        
        for i in range(0, len(summaries)):
            print('Auto summary: {}'.format(h_summaries[i]))
            print()
            print('Gold summary: {}'.format(summaries[i]))
            print()
        
        print(rouge_scores)

    else:
        stories_train, summaries_train, stories_test, summaries_test, word_count= dh.load_stories(file_loc, num=num_)
        embeddings,word_to_int, int_to_word=dh.build_vocab(word_count)
        X_train =dh.convert_text_to_int(stories_train, word_to_int, eos=True)
        Y_train =dh.convert_text_to_int(summaries_train, word_to_int)
        X_test = dh.convert_text_to_int(stories_test, word_to_int) # we don't need EOS, we will loop through these ourselves
        Y_test = dh.convert_text_to_int(summaries_test, word_to_int)
        
        update_check = (len(X_train)//batch_size//per_epoch)-1 
        #Build graph
        train_graph, cost_func, train_op, input_data, targets, lr, keep_prob, summary_length, max_sentence_length, text_length,summary_op = build_graph(word_to_int, 
                embeddings, rnn_size, num_layers, batch_size)

        cwd = os.getcwd()
        checkpoint =os.path.join(cwd,"best_model_short.ckpt")

        with tf.Session(graph = train_graph) as sess:
            log_writer = tf.summary.FileWriter('./logs', train_graph)
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(1, epochs + 1):
                for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(get_batches(Y_train, X_train,batch_size, word_to_int)):
                    start_time = time.time()
                    log_summary,_, loss = sess.run([summary_op, train_op, cost_func], 
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
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f})'.format(epoch_i, 
                            epochs, 
                            batch_i, 
                            len(X_train)//batch_size, 
                            batch_loss/display_step, 
                            batch_time*display_step))
                        batch_loss = 0

                        log_writer.add_summary(log_summary, batch_i)
                    
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
        
                #Reduce learning rate, but not below its minimum value
                learning_rate *= 0.95 
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate
            
                if stop_early == stop:
                    print("Stopping Training.")
                    break 
       
       #Test with test data set after modeling
        avg_summary_length = sum(summaries_lengths)//len(summaries_lengths)
        y_sys = test_model(X_test, Y_test, './best_model.ckpt', train_graph,batch_size, avg_summary_length) # How do you test without knowing training batch_size
        h_summaries = dh.convert_int_to_text(y_sys, int_to_word, word_to_int['<PAD>'])
        rouge_scores = evalhelper.evaluate_rouge_score(h_summaries, summaries_test)
        
        for i in range(0, len(summaries_test)):
            print('Auto summary: {}'.format(h_summaries[i]))
            print()
            print('Gold summary: {}'.format(summaries_test[i]))
            print()
        
        print(rouge_scores)

def test_model(X_test, y_test, model_cpk, test_graph, batch_size, avg_summary_length):

    with tf.Session(graph = test_graph) as sess:
        sess.run(tf.global_variables_initializer())
        loader = tf.train.import_meta_graph(model_cpk + '.meta')
        loader.restore(sess, model_cpk)

        input_data = test_graph.get_tensor_by_name('inputs:0')
        infer_logits = test_graph.get_tensor_by_name('predictions:0')
        text_length = test_graph.get_tensor_by_name('text_sequence_length:0')
        summary_length = test_graph.get_tensor_by_name('summary_sequence_length:0')
        keep_prob = test_graph.get_tensor_by_name('keep_prob:0')
        
        y_sys = []

        for x_ in X_test:
            y_ = sess.run(infer_logits, {input_data: [x_]*batch_size,
                summary_length:[avg_summary_length],
                text_length:[len(x_)]*batch_size,
                keep_prob: 1.0})[0]

            y_sys.append(y_.tolist())
        #compute accuracy measures

    return y_sys 

    
    



if __name__ == "__main__":

    while True:
        opt = input('Please select option below:\n\t1.Train \n\t2.Test\n\t3.Exit\n')
        print('Selection option: {}'.format(opt))
        if int(opt) == 1:
            train_dir = input('Enter directory containing training files: ')
            if not os.path.isdir(train_dir):
                print('Directory: {} does not exist'.format(train_dir))
                continue
            num_files = int(input('Enter number of files to train(minimum = 200): '))
            if num_files is None:
                continue
            batch_size = int(input('Enter batch size for training or <Enter> for default 32: '))

            if batch_size is None or batch_size > num_files:
                print('Batch size has to be less than number of files.')
                batch_size = 32 #Try to find a better wayy
            epochs = int(input('Enter number of epochs or <Enter> for default 60: '))

            if epochs is None:
                print('Using defaults for epochs 60')
                epochs = 60
            main(Test = False, file_loc = train_dir, n = num_files, batch_s = batch_size, epoch_n = epochs)    
        elif int(opt) == 2:
            test_dir = input('Enter directory containing testing files: ')
            if not os.path.isdir(test_dir):
                print('Directory: {} does not exist.'.format(test_dir))
                continue
            num_files = int(input('Enter number of files to test: '))
            main(Test = True, file_loc = test_dir, n = num_files)
        elif int(opt)== 3:
            break
        else:
            print('Unknown option. Try again.')
        
