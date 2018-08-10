from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np
from attention_decoder import attention_decoder


def add_encoder(H,initializer,inputs,sequence_length):
    cell_fw = tf.contrib.rnn.LSTMCell(H, initializer=initializer, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(H, initializer=initializer, state_is_tuple=True)
    (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                        cell_bw, 
                                                                        inputs, 
                                                                        dtype=tf.float32, 
                                                                        sequence_length=sequence_length, 
                                                                        swap_memory=True)
    encoder_outputs = tf.concat(axis=2, values=encoder_outputs) 
    return encoder_outputs, fw_st, bw_st


def add_decoder(hps, inputs,initializer,input_state,encoder_state,encoder_padding):
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=initializer)
    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, input_state, 
                                                                        encoder_state, encoder_padding, 
                                                                        cell, initial_state_attention=False, 
                                                                        pointer_gen=False, use_coverage=False, 
                                                                        prev_coverage=None)

    return outputs, out_state, attn_dists, p_gens, coverage




def reduce_states(hps, fw_st, bw_st,initializer):
    hidden_dim = hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=initializer)
        w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=initializer)
        bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=initializer)
        bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=initializer)

        # Apply linear layer
        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
    return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state



def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class SummarizationModel(object):

    def __init__(self, graph=None, *args, **kwargs):
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self,hps, vocab):
        self._hps = hps
        self._vocab = vocab
        self._vocab_size = vocab.size()

    @with_self_graph
    def BuildCoreGraph(self):
        
        tf.logging.info("Building core graph...")
        self.global_step = tf.train.get_or_create_global_step()
        
        #encoder
        self._enc_batch = tf.placeholder(tf.int32, [self._hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [self._hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [self._hps.batch_size, None], name='enc_padding_mask')

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, 
                                         [self._hps.batch_size, self._hps.max_dec_steps], 
                                         name='dec_batch')
        
        self._target_batch = tf.placeholder(tf.int32, 
                                            [self._hps.batch_size, self._hps.max_dec_steps], 
                                            name='target_batch')

        self._dec_padding_mask = tf.placeholder(tf.float32, 
                                                [self._hps.batch_size, self._hps.max_dec_steps], 
                                                name='dec_padding_mask')

        #initializers
        self._rand_unif_init = tf.random_uniform_initializer(-self._hps.rand_unif_init_mag, 
                                                             self._hps.rand_unif_init_mag, seed=123)
        self._trunc_norm_init = tf.truncated_normal_initializer(stddev=self._hps.trunc_norm_init_std)  
        
        
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [self._vocab_size, self._hps.emb_dim], 
                                        dtype=tf.float32, initializer=self._trunc_norm_init)
            self.emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) 
            self.emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] 


        with tf.variable_scope('encoder'):
            self._enc_states, fw_st, bw_st = add_encoder(self._hps.hidden_dim,self._rand_unif_init,
                                                         self.emb_enc_inputs,self._enc_lens)
            #decoder input
            self._dec_in_state = reduce_states(self._hps,fw_st, bw_st, self._trunc_norm_init)


        with tf.variable_scope('attentionDecoder'):
            decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = add_decoder(self._hps,
                                                                             self.emb_dec_inputs,
                                                                             self._rand_unif_init,
                                                                             self._dec_in_state,
                                                                             self._enc_states,
                                                                             self._enc_padding_mask)

        with tf.variable_scope('projection'):
            w = tf.get_variable('w', [self._hps.hidden_dim, self._vocab_size], 
                                dtype=tf.float32, initializer=self._trunc_norm_init)
            w_t = tf.transpose(w)
            v = tf.get_variable('v', [self._vocab_size], dtype=tf.float32, initializer=self._trunc_norm_init)
            self.vocab_scores = [] 
            for i,output in enumerate(decoder_outputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                self.vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) 

            self.vocab_dists = [tf.nn.softmax(s) for s in self.vocab_scores] 


        with tf.variable_scope('loss'):
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(self.vocab_scores, axis=1), 
                                                          self._target_batch, self._dec_padding_mask)

        tf.summary.scalar('loss', self._loss)
        tf.logging.info("Building core graph...COMPLETE")


    @with_self_graph
    def BuildSamplerGraph(self):
        tf.logging.info("Building Sampler graph...")
        assert len(self.vocab_dists) == 1 
        final_dists = self.vocab_dists[0]
        topk_probs, self._topk_ids = tf.nn.top_k(final_dists, self._hps.beam_size*2) 
        self._topk_log_probs = tf.log(topk_probs)        
        tf.logging.info("Building Sampler graph...COMPLETE")
        

    @with_self_graph
    def BuildTrainGraph(self):
        # Define optimizer and training op
        tf.logging.info("Building train graph...")
        with tf.name_scope("Training"):
            self.train_loss_ = self._loss
            tvars = tf.trainable_variables()
            optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
            gradients = tf.gradients(self.train_loss_, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
            self.train_step_ = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
            tf.summary.scalar('global_norm', global_norm)
        tf.logging.info("Building train graph...COMPLETE")
        

    @with_self_graph
    def summarizeGraph(self):
        self._summaries = tf.summary.merge_all()
        
        
    @with_self_graph
    def runEncoder(self,session,batch):
        feed_dict = {
            self._enc_batch : batch.enc_batch,
            self._enc_lens : batch.enc_lens,
            self._enc_padding_mask : batch.enc_padding_mask
        }
        
        enc_states, dec_in_state, global_step = session.run([self._enc_states, 
                                                self._dec_in_state,
                                                self.global_step], 
                                                feed_dict)
        
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        
        return enc_states, dec_in_state

    
    @with_self_graph        
    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):

        beam_size = len(dec_init_states)
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens]))
        }

        to_return = {
          "ids": self._topk_ids,
          "probs": self._topk_log_probs,
          "states": self._dec_out_state,
          "attn_dists": self.attn_dists
        }

        results = sess.run(to_return, feed_dict=feed) # run the decoder step
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], 
                                                    results['states'].h[i, :]) for i in range(beam_size)]

        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()
        p_gens = [None for _ in range(beam_size)]
        new_coverage = [None for _ in range(beam_size)]
        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage        
