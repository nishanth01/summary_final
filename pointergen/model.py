from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np
from attention_decoder import attention_decoder
import util


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


def add_decoder(hps, inputs,initializer,input_state,encoder_state,encoder_padding,prev_coverage):
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=initializer)
    prev_coverage = prev_coverage
    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, input_state, 
                                                                        encoder_state, encoder_padding, 
                                                                        cell, initial_state_attention=False,  
                                                                        prev_coverage=prev_coverage)

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
        self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [self._hps.batch_size, None], name='enc_batch_extend_vocab')
        self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')        

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

        
        if self._hps.mode=="decode":
            self.prev_coverage = tf.placeholder(tf.float32, [self._hps.batch_size, None], name='prev_coverage') 
        else:    
            self.prev_coverage = None
            
            
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
            self.decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = add_decoder(self._hps,
                                                                             self.emb_dec_inputs,
                                                                             self._rand_unif_init,
                                                                             self._dec_in_state,
                                                                             self._enc_states,
                                                                             self._enc_padding_mask,
                                                                             self.prev_coverage)

        with tf.variable_scope('projection'):
            self.BuildProjectionGraph()


        if self._hps.mode in ['train', 'eval']:    
            with tf.variable_scope('loss'):
                self.BuildLossGraph()
                

            
        tf.logging.info("Building core graph...COMPLETE")


    @with_self_graph
    def BuildProjectionGraph(self):
        tf.logging.info("Building projection graph...")
        w = tf.get_variable('w', 
                            [self._hps.hidden_dim, self._vocab_size],
                            dtype=tf.float32, 
                            initializer=self._trunc_norm_init)

        w_t = tf.transpose(w)
        v = tf.get_variable('v', [self._vocab_size], dtype=tf.float32, initializer=self._trunc_norm_init)

        self.vocab_scores = [] 
        for i,output in enumerate(self.decoder_outputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            self.vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) 

        vocab_distribution = [tf.nn.softmax(s) for s in self.vocab_scores]
        
        with tf.variable_scope('final_distribution'):
            vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_distribution)]
            attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, self.attn_dists)]
            
            extended_vsize = self._vocab.size() + self._max_art_oovs 
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] 

            batch_nums = tf.range(0, limit=self._hps.batch_size) 
            batch_nums = tf.expand_dims(batch_nums, 1) 
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1] 
            batch_nums = tf.tile(batch_nums, [1, attn_len]) 
            indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) 
            shape = [self._hps.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] 

            final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended,attn_dists_projected)]
            self.vocab_dists = final_dists

        tf.logging.info("Building projection graph...COMPLETE")
        

    @with_self_graph
    def BuildLossGraph(self):
        tf.logging.info("Building Loss graph...")
        loss_per_step = [] 
        batch_nums = tf.range(0, limit=self._hps.batch_size) 
        
        for dec_step, dist in enumerate(self.vocab_dists):
            targets = self._target_batch[:,dec_step] 
            indices = tf.stack( (batch_nums, targets), axis=1) 
            gold_probs = tf.gather_nd(dist, indices) 
            losses = -tf.log(gold_probs)
            loss_per_step.append(losses)

        # Apply dec_padding_mask and get loss
        self._loss = util.mask_and_avg(loss_per_step, self._dec_padding_mask)        
        tf.summary.scalar('loss', self._loss)
        
        with tf.variable_scope('coverage_loss'):
            coverage = tf.zeros_like(self.attn_dists[0]) 
            covlosses = [] 
            for a in self.attn_dists:
                covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) 
                covlosses.append(covloss)
                coverage += a 
            self._coverage_loss = util.mask_and_avg(covlosses, self._dec_padding_mask)
            tf.summary.scalar('coverage_loss', self._coverage_loss)
            
        self._total_loss = self._loss + self._hps.cov_loss_wt * self._coverage_loss
        tf.summary.scalar('total_loss', self._total_loss)        
        tf.logging.info("Building Loss graph...COMPLETE")

        
    @with_self_graph
    def BuildTrainGraph(self):
        # Define optimizer and training op
        tf.logging.info("Building train graph...")
        with tf.name_scope("Training"):
            self.train_loss_ = self._total_loss
            tvars = tf.trainable_variables()
            optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
            gradients = tf.gradients(self.train_loss_, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
            self.train_step_ = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
            tf.summary.scalar('global_norm', global_norm)
        tf.logging.info("Building train graph...COMPLETE")

        
    @with_self_graph
    def BuildSamplerGraph(self):
        tf.logging.info("Building Sampler graph...")
        assert len(self.vocab_dists) == 1 
        final_dists = self.vocab_dists[0]
        topk_probs, self._topk_ids = tf.nn.top_k(final_dists, self._hps.beam_size*2) 
        self._topk_log_probs = tf.log(topk_probs)        
        tf.logging.info("Building Sampler graph...COMPLETE")
        

    @with_self_graph
    def summarizeGraph(self):
        tf.logging.info("Building summary graph...")
        self._summaries = tf.summary.merge_all()
        tf.logging.info("Building summary graph...COMPLETE")
        
        
    @with_self_graph
    def runEncoder(self,session,batch):
        feed_dict = {
            self._enc_batch : batch.enc_batch,
            self._enc_lens : batch.enc_lens,
            self._enc_padding_mask : batch.enc_padding_mask,
            self._enc_batch_extend_vocab: batch.enc_batch_extend_vocab,
            self._max_art_oovs : batch.max_art_oovs            
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
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self._enc_batch_extend_vocab : batch.enc_batch_extend_vocab,
            self._max_art_oovs: batch.max_art_oovs,
            self.prev_coverage: np.stack(prev_coverage, axis=0)
        }

        to_return = {
          "ids": self._topk_ids,
          "probs": self._topk_log_probs,
          "states": self._dec_out_state,
          "attn_dists": self.attn_dists,
          "p_gens":self.p_gens,
          "coverage" : self.coverage
        }

        results = sess.run(to_return, feed_dict=feed) # run the decoder step
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], 
                                                    results['states'].h[i, :]) for i in range(beam_size)]

        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()
        
        assert len(results['p_gens'])==1
        p_gens = results['p_gens'][0].tolist()
        
        new_coverage = results['coverage'].tolist()
        assert len(new_coverage) == beam_size        
        
        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage      
    
    
