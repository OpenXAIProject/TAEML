import tensorflow as tf 
import numpy as np 
import argparse
import time
import os
import pdb

from lib.episode_generator import EpisodeGenerator
from lib.networks import ProtoNet, Lrn2evl
from lib.networks import cross_entropy, tf_acc
from config.loader import load_config
#from lib.networks import prototypical_net, taeml_net

def parse_args():
    parser = argparse.ArgumentParser(description='learning to ensemble')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxe', dest='max_iter', default=15000, type=int)
    parser.add_argument('--qs', dest='qsize', default=15, type=int) 
    parser.add_argument('--nw', dest='nway', default=10, type=int) 
    parser.add_argument('--ks', dest='kshot', default=5, type=int)
    parser.add_argument('--sh', dest='show_step', default=100, type=int)
    parser.add_argument('--sv', dest='save_step', default=1000, type=int)
    parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
    parser.add_argument('--data', dest='dataset_dir', default='../datasets')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--dset', dest='dataset_name', default=None)
    parser.add_argument('--name', dest='project_name', default='ProtoNetSingle')
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
    parser.add_argument('--config', dest='config', default='general')
    parser.add_argument('--from_ckpt', dest='from_ckpt', 
            default=False, type=bool)
    parser.add_argument('--ens_name', dest='ens_name', default='taeml')
    parser.add_argument('--trd', dest='transductive', 
            default=False, action='store_true')
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

    models = [[] for _ in range(len(TRAIN_DATASET))]
    loaders = [[] for _ in range(len(TRAIN_DATASET))] 
    with tf.variable_scope(args.project_name):
        for tn, tset in enumerate(TRAIN_DATASET):
            models[tn] = ProtoNet(tset, nway, kshot, qsize, isTr=False, 
                    input_dict=input_dict)
            tnvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                    scope=args.project_name + '/' + tset + '/*')
            loaders[tn] = tf.train.Saver(tnvars)
        
        embeds = [models[tn].outputs['embedding'] for tn in range(len(TRAIN_DATASET))]
        preds = [models[tn].outputs['pred'] for tn in range(len(TRAIN_DATASET))]
    
        embeds = tf.convert_to_tensor(embeds) # (M, Q, N, H)
        preds = tf.convert_to_tensor(preds)
        
        # preds.shape: (M, NQ, N)
        if args.transductive:
            fe = Lrn2evl(args.ens_name, embeds, nway) # (M,1)
            fe = tf.reshape(fe, [-1,1,1]) # (M,1,1)
        else:
            fe = Lrn2evl(args.ens_name, embeds, nway, 
                    transduction=False) # (MNQ,1)
            fe = tf.reshape(fe, [len(TRAIN_DATASET), nway*qsize, 1]) # (M,NQ,1)

        pe = tf.reduce_sum(fe * preds, axis=0) 
        pe = tf.nn.softmax(pe)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=args.project_name + '/' + args.ens_name + '/*')

        acc = tf_acc(pe, qy_ph)
        loss = cross_entropy(pe, qy_ph) 
        opt = tf.train.AdamOptimizer(lr_ph) 
        train_op = opt.minimize(loss, var_list=train_vars) 

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if not args.from_ckpt:
        for tn, tset in enumerate(TRAIN_DATASET):
            saved_loc = os.path.join(args.model_dir,
                    args.project_name,
                    tset + '.ckpt')
            loaders[tn].restore(sess, saved_loc)
            print ('model_{} restored from {}'.format(tn, saved_loc))
    else:
        out_loc = os.path.join(args.model_dir,  
                args.project_name,
                args.ens_name + '.ckpt')
        saver.restore(sess, out_loc)
    
    ep_gen = EpisodeGenerator(args.dataset_dir, 'val', config)
    ep_test = EpisodeGenerator(args.dataset_dir, 'test', config)
    avger = np.zeros([4])
    np.set_printoptions(precision=3, suppress=False)
    for i in range(1, args.max_iter+1): 
        stt = time.time()
        lr = args.lr if i < 0.7 * args.max_iter else args.lr*.1
        sx, sy, qx, qy = ep_gen.get_episode(nway, kshot, qsize, printname=False)
        bedt = time.time() - stt
        fd = {sx_ph: sx, qx_ph: qx, qy_ph: qy, lr_ph: lr}
        
        loss_val, acc_val, W_val, _ = sess.run([loss, acc, fe, train_op], fd)
        avger += [loss_val, acc_val, bedt, time.time() - stt] 

        if i % args.show_step == 0 and i != 0: 
            avger /= args.show_step
            print ('step : {:8d}/{}  loss: {:.3f}   |   acc: {:.3f}  | batchtime: {:.2f}  | in {:.2f} secs'\
                    .format(i, args.max_iter, avger[0], 
                        avger[1], avger[2]*args.show_step, avger[3]*args.show_step))
            avger[:] = 0

#            print ('+'*30)
#            print (TRAIN_DATASET)
#            for _ in range(5):
#                sx, sy, qx, qy = ep_test.get_episode(nway, kshot, qsize, printname=True, 
#                        dataset_name='miniImagenet')
#                fd = {sx_ph: sx, qx_ph: qx, qy_ph: qy, lr_ph: lr}
#                p, p1, p2, p3 = sess.run([fe, preds, pe, acc], fd)
#                print (np.mean(p, axis=(1,2)))
#            print ('+'*30)

        if i % args.save_step == 0 and i != 0: 
            out_loc = os.path.join(args.model_dir,  #models/ 
                    args.project_name,
                    args.ens_name + '.ckpt')
            saver.save(sess, out_loc)
            print ('saved at {}'.format(out_loc))
