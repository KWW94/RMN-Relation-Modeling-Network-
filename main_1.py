import tensorflow as tf
import numpy as np
import h5py, random, csv, gzip, time 
import pickle
from util import *
from layers import RMN
import os
import datetime
span_data, span_size, wmap, cmap, bmap = load_data('data/relationships.csv.gz', 'data/metadata.pkl')

with open('data/glove.We', 'rb') as f:
    We = pickle.load(f, encoding='latin1').astype('float32')

norm_We = We / np.linalg.norm(We, axis=1)[:, None]
We = np.nan_to_num(norm_We)

descriptor_log = 'models/descriptors.log'
trajectory_log = 'models/trajectories.log'

# embedding/hidden dimensionality
d_word = We.shape[1]
d_char = 50
d_book = 50
d_hidden = 50

# number of descriptors
num_descs = 30

# number of negative samples per relationship
num_negs = 50

# word dropout probability
p_drop = 0.75

n_epochs = 90
lr = 0.001
eps = 1e-6

num_chars = len(cmap)
num_books = len(bmap)
num_traj = len(span_data)
len_voc = len(wmap)
revmap = {}

for w in wmap:
    revmap[wmap[w]] = w
	

epoch_loss_list = []

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    session_conf = tf.ConfigProto( allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        print ('compiling...')
        RMN = RMN( d_word, d_char, d_book, d_hidden, len_voc, num_descs, num_chars, num_books, span_size, We, eps=eps, lr=lr, negs=num_negs )
        print ('done compiling, now training...')
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(RMN.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        # training loop
        def train_step(chars, book, curr, cm, drop_mask, ns, nm, length):
            """
            A single training step
            """
            RMN.length = length
            # print (RMN.length)
            feed_dict = {RMN.input_chars : chars, 
                         RMN.input_book : book, 
                         RMN.input_spans : curr, 
                         RMN.input_currmask : cm, 
                         RMN.input_dropmask : drop_mask , 
                         RMN.input_neg : ns,
                         RMN.input_negmask : nm}
            
            _, step, loss, desciptor_R, input_recu = sess.run(
                [train_op, global_step, RMN.loss, RMN.desciptor_R, RMN.input_recu],
                feed_dict)
            
            		
            return loss, desciptor_R, input_recu
        def traj_step(chars, book, curr, cm):
            """
            A single training step
            """
            # print (RMN.length)
            feed_dict = {RMN.input_chars : chars, 
                         RMN.input_book : book, 
                         RMN.input_spans : curr, 
                         RMN.input_dropmask : cm
						 }
            
            input_recu = sess.run(
                [ RMN.input_recu],
                feed_dict)
            
            		
            return input_recu

        min_cost = float('inf')
        random.shuffle(span_data)
        for epoch in range(n_epochs):
            cost = 0.

            start_time = time.time()
            print ('start epoch ', epoch)
            for book, chars, curr, cm, in span_data[:100]:
                c1, c2 = [cmap[c] for c in chars]
                bname = bmap[book[0]]
                # print (bname,c1,c2)
                for index in range(len(curr)):
                    ns, nm = generate_negative_samples(num_traj, span_size, num_negs, span_data)
                
                    drop_mask = (np.random.rand(*(cm[index].shape)) < (1 - p_drop)).astype('float32')
                    drop_mask *= cm[index]
                
                    ex_cost, desciptor_R, traj = train_step(chars, book, np.expand_dims(curr[index],axis=0), np.expand_dims(cm[index],axis=0), np.expand_dims(drop_mask,axis=0), ns, nm, len(curr))
                    cost += ex_cost
                    # print (traj.shape)
                    # traj_writer.writerow([bname, c1, c2, index] + list(traj[0]) )
                    # tlog.flush()
                    # print ('trajectories %d:' % ind)

                    time_str = datetime.datetime.now().isoformat()
                    
                current_step = tf.train.global_step(sess, global_step)
                print(time_str,":", "step : ",current_step, "book : " ,bname,'\n', "loss : " , ex_cost[0][0])

            end_time = time.time() 

            # compute nearest neighbors of descriptors
            R = desciptor_R
            log = open(descriptor_log, 'w')
            for ind in range(len(R)):
                desc = R[ind] / np.linalg.norm(R[ind])
                sims = We.dot(desc.T)
                ordered_words = np.argsort(sims)[::-1]
                desc_list = [ revmap[w] for w in ordered_words[:10]]
                log.write(' '.join(desc_list) + '\n')
                # print ('descriptor %d:' % ind)
                # print (desc_list)
            log.flush()
            log.close()

            # save relationship trajectories
            print ('writing trajectories...')
            tlog = open(trajectory_log, 'w')
            traj_writer = csv.writer(tlog)
            traj_writer.writerow(['Book', 'Char 1', 'Char 2', 'Span ID'] + ['Topic ' +str(i) for i in range(num_descs)])
            for book, chars, curr, cm in span_data[:100]:
                for index in range(len(curr)):
                    c1, c2 = [cmap[c] for c in chars]
                    bname = bmap[book[0]]
                    # feed unmasked inputs to get trajectories
                    traj = traj_step(chars, book, np.expand_dims(curr[index],axis=0), np.expand_dims(cm[index],axis=0))
                    # print (np.array(traj).shape)
                    for ind in range(len(traj[0][0])):
                        step = traj[ind]
                        # print (np.array(step).shape)
                        traj_writer.writerow([bname, c1, c2, index] + list(step[0]) )
                        tlog.flush()
                        print (bname + 'trajectories %d: ' % index )		
                    #print (traj[0][0])			
            tlog.flush()
            tlog.close()
			
            print ('done with epoch: ', epoch, ' cost =',cost / len(span_data), 'time: ',end_time-start_time)
            epoch_loss_list.append(cost / len(span_data))
            np.savez('epoch_loss_list.npz',loss=epoch_loss_list)
            saver.save(sess, 'save_model/RMN_model')
            