import queue as Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data
from batch import Example,Batch


class Batcher(object):

    BATCH_QUEUE_MAX = 1000000 

    def __init__(self, data_path, vocab, hps):

        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = hps.single_pass

        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
        self._num_batch_q_threads = 1  # just one thread to batch examples
        self._bucketing_cache_size = 1 
        self._finished_reading = False # this will tell us when we're finished reading the dataset

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
            self._batch_q_threads = []
            
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()
            
            
    def next_batch(self):
        if self._batch_queue.qsize() == 0:
            tf.logging.warning('Bucket input queue is empty when calling next_batch.')
            if self._finished_reading:
                tf.logging.info("Finished reading dataset.")
                return None

        batch = self._batch_queue.get() 
        return batch            

    
    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))
        while True:
            try:
                (article, abstract) = next(input_gen) 
            except StopIteration: # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                self._finished_reading = True
                break

            abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] 
            example = Example(article, abstract_sentences, self._vocab, self._hps) 
            self._example_queue.put(example) 

            
    def fill_batch_queue(self):
        while True:
            ex = self._example_queue.get()
            b = [ex for _ in range(self._hps.batch_size)]
            self._batch_queue.put(Batch(b, self._hps, self._vocab))            

            
    def text_generator(self, example_generator):
        while True:
            e = next(example_generator) 
            try:
                article_text = e.features.feature['article'].bytes_list.value[0].decode() 
                abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode() 
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
                
            if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
                tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield (article_text, abstract_text)                
                
                