import tensorflow as tf 
import numpy as np 
import argparse
import os
from lib.episode_generator import EpisodeGenerator
from lib.networks import cross_entropy, tf_acc
from lib.networks import ProtoNet
from config.loader import load_config
import pdb

np.random.seed(0)
def parse_args():
    parser = argparse.ArgumentParser(description='ensemble test')
    parser.add_argument('--name', dest='project_name', default='ProtoNetSingle', type=str)
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxi', dest='max_iter', default=600, type=int)
    parser.add_argument('--qs', dest='qsize', default=15, type=int) 
    parser.add_argument('--nw', dest='nway', default=10, type=int) 
    parser.add_argument('--ks', dest='kshot', default=5, type=int)
    parser.add_argument('--sh', dest='show_step', default=100, type=int)
    parser.add_argument('--data', dest='dataset_dir', default='../datasets')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--gpufrac', dest='gpufraction', default=0.9, type=float)
    parser.add_argument('--save_cache', dest='save_cache', default=False)
    parser.add_argument('--config', dest='config', default='general')
    parser.add_argument('--gpuf', dest='gpu_frac', default=0.44)
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
    config = load_config(args.config)
    TRAIN_DATASET = config['TRAIN_DATASET'].copy()

    sx_ph = tf.placeholder(tf.float32, [nway*kshot,84,84,3])
    qx_ph = tf.placeholder(tf.float32, [nway*qsize,84,84,3])
    qy_ph = tf.placeholder(tf.float32, [nway*qsize,nway]) 
    lr_ph = tf.placeholder(tf.float32) 
    
    input_dict = {'sx': sx_ph, 'qx': qx_ph, 'qy': qy_ph}
    
#    TRAIN_DATASET.append('multiple') # protonet trained on multiple datasets
#    print (TRAIN_DATASET)

    models = [[] for _ in range(len(TRAIN_DATASET))]
    loaders = [[] for _ in range(len(TRAIN_DATASET))]
    with tf.variable_scope(args.project_name):
        for tn, tset in enumerate(TRAIN_DATASET):
            models[tn] = ProtoNet(tset, nway, kshot, qsize, isTr=False,
                    input_dict=input_dict)
            tnvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                    scope=args.project_name + '/' + tset + '/*')
            loaders[tn] = tf.train.Saver(tnvars)
        preds = [models[tn].outputs['pred'] for tn in range(len(TRAIN_DATASET))]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    for tn, tset in enumerate(TRAIN_DATASET):
        saved_loc = os.path.join(args.model_dir,
                args.project_name,
                tset + '.ckpt') 
        loaders[tn].restore(sess, saved_loc)
        print ('model_{} restored from {}'.format(tn, saved_loc))
    
    ep_gen = EpisodeGenerator(args.dataset_dir, 'test', config)
    target_dataset = ep_gen.dataset_list
    #target_dataset = ['miniImagenet']
    for tdi, dset in enumerate(target_dataset):
        print ('==========TARGET : {}=========='.format(dset)) 
        results = [[] for _ in range(len(TRAIN_DATASET))]
        for i in range(args.max_iter):
            sx, sy, qx, qy = ep_gen.get_episode(nway, kshot, qsize,
                    dataset_name=dset)
            fd = {sx_ph: sx, qx_ph: qx, qy_ph: qy}
            ps = sess.run(preds, fd)
            emb = sess.run(models[0].outputs['embedding'], fd)
            for pn, p in enumerate(ps): 
                temp_acc = np.mean(np.argmax(p, 1) == np.argmax(qy, 1))
                results[pn].append(temp_acc)
        
        for i in range(len(TRAIN_DATASET)):
            print ('model trained on {:10s} - Acc {:.3f} ({:.3f})'.format(\
                    TRAIN_DATASET[i], 
                    np.mean(results[i]),
                    #np.std(results[i]) * 100. * 1.96 / np.sqrt(args.max_iter)))
                    np.std(results[i])))
        #print ('Ensemble Avg model - Acc {:.3f}'.format(acc_results[-1]))



        

                
        






#
