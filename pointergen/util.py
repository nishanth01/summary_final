import tensorflow as tf
import time
import os



def load_ckpt(saver, session, hps, ckpt_dir="train"):
    while True:
        try:
            latest_filename = "checkpoint"# if ckpt_dir=="eval" else hps.model_name
            ckpt_dir = os.path.join(hps.log_root,ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(session, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception as e:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            raise e

            
def running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
        running_avg_loss = min(running_avg_loss, 12)  # clip
        loss_sum = tf.Summary()
        tag_name = 'running_avg_loss/decay=%f' % (decay)
        loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
        summary_writer.add_summary(loss_sum, step)
        #tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss



def mask_and_avg(values, padding_mask):
    dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
    values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average
