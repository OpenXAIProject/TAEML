import tensorflow as tf 
import numpy as np 
import argparse
import time
import os
from lib.episode_generator import EpisodeGenerator, TRAIN_DATASET
from lib.networks import Baseline_CNN, cross_entropy, tf_acc
from lib.networks import prototypical_net, taeml_net, taeml_net_nontrans

np.random.seed(0)
def parse_args():
    parser = argparse.ArgumentParser(description='learning to ensemble')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxe', dest='max_iter', default=25, type=int)
    parser.add_argument('--qs', dest='qsize', default=3, type=int) 
    parser.add_argument('--nw', dest='nway', default=10, type=int) 
    parser.add_argument('--ks', dest='kshot', default=5, type=int)
    parser.add_argument('--sh', dest='show_step', default=100, type=int)
    parser.add_argument('--sv', dest='save_step', default=1000, type=int)
    parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
    parser.add_argument('--data', dest='dataset_dir', default='../datasets')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--dset', dest='dataset_name', default=None)  
    parser.add_argument('--name', dest='model_name', default='baselearner_5s')  
    parser.add_argument('--gpufrac', dest='gpufraction', default=0.95, type=float)
    parser.add_argument('--save_cache', dest='save_cache', default=False)
    parser.add_argument('--ens_name', dest='ens_name', default='taeml_5s')
    parser.add_argument('--trd', dest='transductive', default=False)
    parser.add_argument('--norm', dest='norm', default=True, type=bool, 
            help="normalized prototypical network")
    args = parser.parse_args()
    return args

if __name__=='__main__': 
    args = parse_args() 
    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    nway = args.nway
    kshot = args.kshot
    qsize = args.qsize 

    sx_ph = tf.placeholder(tf.float32, [nway*kshot,84,84,3])
    qx_ph = tf.placeholder(tf.float32, [nway*qsize,84,84,3])
    qy_ph = tf.placeholder(tf.float32, [nway*qsize,nway]) 

    
    models = [[] for _ in range(len(TRAIN_DATASET))]
    preds = [[] for _ in range(len(TRAIN_DATASET))]
    embeds = [[] for _ in range(len(TRAIN_DATASET))]
    with tf.variable_scope(args.model_name):
        for tn, tset in enumerate(TRAIN_DATASET):
            models[tn] = Baseline_CNN(tset)
            pdict = prototypical_net(models[tn], sx_ph, qx_ph, nway, 
                    isTr=False, trainable=False, normalized=args.norm)
            preds[tn] = pdict['pred'] 
            embeds[tn] = pdict['embedding']
    embeds = tf.convert_to_tensor(embeds)
    
    if args.transductive:
        pemb = taeml_net(args.ens_name, embeds, nway) 
        pemb = tf.reshape(pemb, [-1,1,1])
    else:
        pemb = taeml_net_nontrans(args.ens_name, embeds, nway) 
        pemb = tf.reshape(pemb, [len(TRAIN_DATASET), nway*qsize, 1])
    
    ens_pred = tf.reduce_sum(pemb * preds, axis=0) 

    acc = tf_acc(ens_pred, qy_ph)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    load_loc = os.path.join(args.model_dir, 
            args.model_name, 
            args.ens_name + '.ckpt')
    saver.restore(sess, load_loc)
    
    ep_gen = EpisodeGenerator(args.dataset_dir, 'test')
    target_dataset = ep_gen.dataset_list
    means = np.zeros([len(TRAIN_DATASET)+2, len(target_dataset)])
    stds  = np.zeros([len(TRAIN_DATASET)+2, len(target_dataset)])
    for tdi, dset in enumerate(target_dataset):
        print ('==========TARGET : {}=========='.format(dset)) 
        temp_results = [[] for _ in range(len(TRAIN_DATASET)+2)]
        for i in range(args.max_iter):
            sx, sy, qx, qy = ep_gen.get_episode(nway, kshot, qsize,
                    dataset_name=dset)
            fd = {sx_ph: sx, qx_ph: qx, qy_ph: qy}
            ps, p_acc, p_W = sess.run([preds, acc, pemb], fd)
            prediction = np.argmax(ps, axis=2) # (5, 150)

            for pn, p in enumerate(ps): 
                temp_p = np.argmax(p, 1)
                temp_results[pn].append(np.mean(temp_p == np.argmax(qy, 1)))
            temp_results[pn+1].append(np.mean(np.argmax(np.mean(ps,0),1)\
                    == np.argmax(qy,1)))
            temp_results[pn+2].append(p_acc)

        for i in range(len(TRAIN_DATASET)):
            print ('model trained on {:10s} - Acc {:.3f}'.format(\
                    TRAIN_DATASET[i], np.mean(temp_results[i])))
            means[i, tdi] = np.mean(temp_results[i])
            stds[i, tdi] = np.std(temp_results[i])
    

        print ('Ensemble Avg model - Acc {:.3f}'.format(np.mean(temp_results[i+1])))
        print ('Learning2Ens model - Acc {:.3f}'.format(np.mean(temp_results[i+2])))
        means[i+1,tdi] = np.mean(temp_results[i+1])
        stds[i+1,tdi] = np.std(temp_results[i+1])
        means[i+2,tdi] = np.mean(temp_results[i+2])
        stds[i+2,tdi] = np.std(temp_results[i+2])


    # Latex stlye reults 
    # 1. no std version
#    dlabels = TRAIN_DATASET+['Uniform Averaging Ensemble', 'TAEML']
#    out_filename = 'logs/nostd/' + '{}.n_{}.k_{}.q_{}.trans.txt'.format(\
#            nway, kshot, nway*qsize, args.transductive)
#    out_text = ''
#    for i in range(len(dlabels)):
#        txt = '{:30s}    &{:.3f}      &{:.3f}      &{:.3f}      &{:.3f}     &{:.3f}  \\\\'.\
#                format(dlabels[i], means[i,0], means[i,1],
#                    means[i,2], means[i,3], means[i,4])
#        out_text += txt + '\n'
#    f = open(out_filename, 'wt')
#    f.write(out_text)
#    print ('saved at {}'.format(out_filename))
    
    
    # 2. std version
    dlabels = TRAIN_DATASET+['Uniform Averaging Ensemble', 'TAEML']
    out_filename = 'logs/std/' + '{}.n_{}.k_{}.q_{}.trans.txt'.format(\
            nway, kshot, nway*qsize, args.transductive)
    out_text = ''
    ci = stds * 1.96 / np.sqrt(args.max_iter)
    for i in range(len(dlabels)):
        txt = '{:30s}    &{:.3f} (\\pn{:.3f})      &{:.3f} (\\pn{:.3f})      &{:.3f} (\\pn{:.3f})      &{:.3f} (\\pn{:.3f})     &{:.3f} (\\pn{:.3f}) \\\\'.\
                format(dlabels[i], means[i,0], ci[i,0], means[i,1], ci[i,1],
                    means[i,2], ci[i,2], means[i,3], ci[i,3], means[i,4], ci[i,4])
        out_text += txt + '\n'
    f = open(out_filename, 'wt')
    f.write(out_text)
    print ('saved at {}'.format(out_filename))





                #
