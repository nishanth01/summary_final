import tensorflow as tf
import time
import os
import glob
import random
import struct
import csv
import batch
import data
import model

from batch import Example,Batch
from data import Vocab
from model import SummarizationModel
import util
import numpy as np


from collections import namedtuple
from tensorflow.python import debug as tf_debug
from tensorflow.core.example import example_pb2
import json, re, shutil, sys
import collections, itertools


def training_init(hps):
    vocab = Vocab(hps.vocab_path, hps.vocab_size)
    batches = get_data(hps,vocab,hps.data_path)
    train_dir = os.path.join(hps.log_root, "model")
    if not os.path.exists(train_dir): os.makedirs(train_dir)    
    
    tf.reset_default_graph()
    model_params = dict(hps=hps, 
                        vocab=vocab)

    lm = SummarizationModel(**model_params)
    lm.BuildCoreGraph()
    lm.BuildTrainGraph()
    lm.summarizeGraph()
    
    return lm,vocab,batches,train_dir


def runEvalStep(lm,session,batch):

    feed_dict = {
        lm._enc_batch : batch.enc_batch,
        lm._enc_lens : batch.enc_lens,
        lm._enc_padding_mask : batch.enc_padding_mask,
        lm._dec_batch : batch.dec_batch,
        lm._target_batch : batch.target_batch,
        lm._dec_padding_mask : batch.dec_padding_mask, 
        lm._enc_batch_extend_vocab: batch.enc_batch_extend_vocab,
        lm._max_art_oovs: batch.max_art_oovs            
    }

    return_dict = {
        'summaries': lm._summaries,
        'loss': lm.train_loss_,
        'global_step': lm.global_step,
        'coverage_loss':lm._coverage_loss
    }

    results = session.run(return_dict, feed_dict)
    return results
    
    
def runTrainStep(lm,session,batch):

    feed_dict = {
        lm._enc_batch : batch.enc_batch,
        lm._enc_lens : batch.enc_lens,
        lm._enc_padding_mask : batch.enc_padding_mask,
        lm._dec_batch : batch.dec_batch,
        lm._target_batch : batch.target_batch,
        lm._dec_padding_mask : batch.dec_padding_mask, 
        lm._enc_batch_extend_vocab: batch.enc_batch_extend_vocab,
        lm._max_art_oovs: batch.max_art_oovs            
    }

    return_dict = {
        'train_op': lm.train_step_,
        'summaries': lm._summaries,
        'loss': lm.train_loss_,
        'global_step': lm.global_step,
        'coverage_loss':lm._coverage_loss
    }

    results = session.run(return_dict, feed_dict)
    return results



def run_epoch(lm,session,batches,summary_writer,train_dir,train_step,saver,hps,best_loss,avg_loss,save_all=True):
    print_interval = 10.0
    total_batches = 0
    total_words = 0     

    start_time = time.time()
    tick_time = start_time  # for showing status
    i = 0
    exception_count = 0
    init_exception_count = 0
    batches_skipped = 0

    for batch in batches:    
        try:
            results = runTrainStep(lm,session,batch)

            loss = results['loss']
            coverage_loss = results['coverage_loss']
            if not np.isfinite(loss):
                raise Exception("Loss is not finite.")        

            summaries = results['summaries'] 
            train_step = results['global_step'] 
            summary_writer.add_summary(summaries, train_step) 

            avg_loss = util.running_avg_loss(np.asscalar(loss), avg_loss, summary_writer, train_step)
            if best_loss is None or avg_loss < best_loss:
                #saver.save(session, train_dir, global_step=train_step, latest_filename='checkpoint_best')
                best_loss = avg_loss            


            total_batches = i + 1
            total_words += len(batch.original_articles)
            i = i + 1

            if (time.time() - tick_time >= print_interval):
                avg_wps = total_words / (time.time() - start_time)
                print("    [batch {:d}]: seen {:d} examples : {:.1f} eps, Loss: {:.3f}, Avg loss: {:.3f}, Best loss: {:.3f}, cov loss: {:.3f}".format(i, total_words, avg_wps, loss,avg_loss,best_loss,coverage_loss))
                tick_time = time.time()  # reset time ticker     
                if save_all:
                    saver.save(session, train_dir, global_step=train_step, latest_filename='checkpoint')
                    
                    

            if train_step % 100 == 0: 
                summary_writer.flush()                

        except Exception as e:
            if(exception_count <= 10):
                print(f'    [EXCEPTION]: ',str(e), '; Restoring model params')
                exception_count = exception_count + 1
                batches_skipped = batches_skipped + 1
                util.load_ckpt(saver, session,hps,hps.log_root)
                continue
            else:
                print('    [EXCEPTION LIMIT EXCEEDED]: Batches skipped:',batches_skipped,'; Error : ',str(e))
                raise e
    time_total = pretty_timedelta(since=start_time)
    print(f"    [END] Training complete: Total examples : {total_words}; Total time: {time_total}")
    return avg_loss,best_loss,train_step


def get_data(hps,vocab,data_path):
    tf.logging.info('Fetching data..') 
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)
    filelist = glob.glob(data_path) 
    inputs = []
    total_examples = 0
    total_batches = 0    
    for f in filelist:
        reader = open(f, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break 
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)
            try:
                article_text = e.features.feature['article'].bytes_list.value[0].decode() 
                if len(article_text)==0: 
                    #tf.logging.warning('Found an example with empty article text. Skipping it.')
                    pass
                else:
                    abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode()
                    abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract_text)]
                    example = Example(article_text,abstract_sentences,vocab,hps)
                    inputs.append(example)       
                    total_examples = total_examples + 1
            except ValueError:
                #tf.logging.error('Failed to get article or abstract from example')            
                continue
    batches = []
    tf.logging.info('Creating batches..') 
    for i in range(0, len(inputs), hps.batch_size):
        batches.append(Batch(inputs[i:i + hps.batch_size],hps,vocab)) 
        total_batches = total_batches + 1
        
    tf.logging.info('[TOTAL Batches]  : %i',total_batches) 
    tf.logging.info('[TOTAL Examples] : %i',total_examples) 
    tf.logging.info('Creating batches..COMPLETE') 
    return batches


def get_decode_data(hps,vocab,data_path,randomize=False):
    tf.logging.info('Fetching data..') 
    filelist = glob.glob(data_path) 
    inputs = []
    total_examples = 0
    total_batches = 0    
    for f in filelist:
        reader = open(f, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break 
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)
            try:
                article_text = e.features.feature['article'].bytes_list.value[0].decode() 
                if len(article_text)==0: 
                    #tf.logging.warning('Found an example with empty article text. Skipping it.')
                    pass
                else:
                    abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode()
                    abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract_text)]
                    example = Example(article_text,abstract_sentences,vocab,hps)
                    inputs.append(example)       
                    total_examples = total_examples + 1
            except ValueError:
                #tf.logging.error('Failed to get article or abstract from example')            
                continue
    batches = []
    tf.logging.info('Creating batches..') 
    if randomize:
        random.shuffle(inputs)
        example = inputs[0]
        b = [example for _ in range(hps.beam_size)]
        batches.append(Batch(b,hps,vocab)) 
        total_batches = 1
        total_examples = 1
    else:
        for i in range(0, len(inputs)):
            b = [inputs[i] for _ in range(hps.beam_size)]
            batches.append(Batch(b,hps,vocab)) 
            total_batches = total_batches + 1
        
    tf.logging.info('[TOTAL Batches]  : %i',total_batches) 
    tf.logging.info('[TOTAL Examples] : %i',total_examples) 
    tf.logging.info('Creating batches..COMPLETE') 
    return batches


def get_specific_example(hps,vocab,example_number):
    
    file_id, number = divmod(example_number,1000)
    path = '/home/ubuntu/W266/final_0/W266_Final/data/final_chunked/validation_%03d.bin'%  file_id
    print(f'Fetching example {number} from: {path}') 
    filelist = glob.glob(path) 
    inputs = []
    total_examples = 0
    total_batches = 0    
    for f in filelist:
        reader = open(f, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break 
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)
            try:
                article_text = e.features.feature['article'].bytes_list.value[0].decode() 
                if len(article_text)==0: 
                    #tf.logging.warning('Found an example with empty article text. Skipping it.')
                    pass
                else:
                    abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode()
                    abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract_text)]
                    example = Example(article_text,abstract_sentences,vocab,hps)
                    inputs.append(example)       
                    total_examples = total_examples + 1
            except ValueError:
                #tf.logging.error('Failed to get article or abstract from example')            
                continue
    batches = []
    tf.logging.info('Creating batches..') 
    example = inputs[number]
    b = [example for _ in range(hps.beam_size)]
    batches.append(Batch(b,hps,vocab)) 
    total_batches = 1
    total_examples = 1
        
    tf.logging.info('[TOTAL Batches]  : %i',total_batches) 
    tf.logging.info('[TOTAL Examples] : %i',total_examples) 
    tf.logging.info('Creating batches..COMPLETE') 
    return batches



def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)