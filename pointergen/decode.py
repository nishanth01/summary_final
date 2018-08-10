import os
import time
import tensorflow as tf
import beam_search
import data
import json
import logging
import numpy as np
import util

from rouge import Rouge

SECS_UNTIL_NEW_CKPT = 60


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def print_results(article, abstract, decoded_output):
    print("---------------------------------------------------------------------------")
    tf.logging.info('ARTICLE:  %s', article)
    tf.logging.info('REFERENCE SUMMARY: %s', abstract)
    tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
    print("---------------------------------------------------------------------------")
            

        
class BeamSearchDecoder(object):
    
    def __init__(self, model, session, vocab, hps, saver):
        self._model = model
        self._vocab = vocab
        self._saver = saver       
        self._session = session
        self._hps = hps
        self.rouge_data = []
        ckpt_path = util.load_ckpt(self._saver, self._session,self._hps,self._hps.cp_dir)
        self.setup_dir()
        

        
    def setup_dir(self):    
        self._decode_dir = os.path.join(self._hps.log_root, "decode")
        if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

        self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
        if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)            
        
        
    def decode(self,batches):
        counter = 0
        for batch in batches:
            if(counter < 10):
                original_article = batch.original_articles[0]  
                original_abstract = batch.original_abstracts[0]
                original_abstract_sents = batch.original_abstracts_sents[0]

                article_withunks = data.show_art_oovs(original_article, self._vocab) 
                abstract_withunks = data.show_abs_oovs(original_abstract, 
                                                       self._vocab, 
                                                       batch.art_oovs[0])

                best_hypothesis = beam_search.run_beam_search(self._session, 
                                                              self._model, 
                                                              self._vocab, 
                                                              batch,
                                                              self._hps)


                output_ids = [int(t) for t in best_hypothesis.tokens[1:]]
                decoded_words = data.outputids2words(output_ids, self._vocab,batch.art_oovs[0])
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words

                decoded_output = ' '.join(decoded_words) # single string

                self.write_for_rouge(original_abstract_sents, decoded_words, counter) 
                counter += 1     
            else:
                break
                
        self.rouge_eval()
            
            
    def decodeOneSample(self,batches):
        
        batch = batches[0]
        original_article = batch.original_articles[0]  
        original_abstract = batch.original_abstracts[0]
        original_abstract_sents = batch.original_abstracts_sents[0]

        article_withunks = data.show_art_oovs(original_article, self._vocab) 
        abstract_withunks = data.show_abs_oovs(original_abstract, 
                                               self._vocab, 
                                               batch.art_oovs[0])

        best_hypothesis = beam_search.run_beam_search(self._session, 
                                                      self._model, 
                                                      self._vocab, 
                                                      batch,
                                                      self._hps)


        output_ids = [int(t) for t in best_hypothesis.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self._vocab,batch.art_oovs[0])
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        decoded_output = ' '.join(decoded_words) # single string


        print_results(article_withunks, abstract_withunks, decoded_output)    
        self.write_for_attnvis(article_withunks, abstract_withunks, 
                               decoded_words, best_hypothesis.attn_dists, 
                               best_hypothesis.p_gens) 
                
    def rouge_eval(self):
        
        hyps, refs = map(list, zip(*[[d['hyp'], d['ref']] for d in self.rouge_data]))
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)    
        
        print('======================================================================')
        print('======================================================================')
        print('ROUGE SCORES')
        print(scores)
        print('======================================================================')
        print('======================================================================')
                
                
    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        article_lst = article.split() # list of words
        decoded_lst = decoded_words # list of decoded words
        to_write = {
                'article_lst': [make_html_safe(t) for t in article_lst],
                'decoded_lst': [make_html_safe(t) for t in decoded_lst],
                'abstract_str': make_html_safe(abstract),
                'attn_dists': attn_dists,
                'p_gens': p_gens
        }
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        tf.logging.info('Wrote visualization data to %s', output_fname)       
        
        
    def write_for_rouge(self, reference_sents, decoded_words, ex_index):
        decoded_sents = []
        data = {}
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError: # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
                
            sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx+1:] # everything else
            decoded_sents.append(' '.join(sent))

        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        data['hyp'] = ' '.join(decoded_sents)
        data['ref'] = ' '.join(reference_sents)
        self.rouge_data.append(data)

        # Write to file
#         ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
#         decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

#         with open(ref_file, "w") as f:
#             for idx,sent in enumerate(reference_sents):
#                 f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")

#         with open(decoded_file, "w") as f:
#             for idx,sent in enumerate(decoded_sents):
#                 f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

        tf.logging.info("Added %i to file" % ex_index)
            