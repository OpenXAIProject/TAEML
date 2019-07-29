import numpy as np 
import tensorflow as tf 
import argparse
import time
import os
import pdb

from lib.episode_generator import EpisodeGenerator
from lib.networks import ProtoNet 
from config.loader import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='protonet')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxe', dest='max_epoch', default=100, type=int)
    parser.add_argument('--qs', dest='qsize', default=15, type=int)
    parser.add_argument('--nw', dest='nway', default=10, type=int)
    parser.add_argument('--ks', dest='kshot', default=1, type=int)
    parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
    parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
    parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
    parser.add_argument('--data', dest='dataset_dir', default='../datasets')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--dset', dest='dataset_name', default='miniImagenet')
    parser.add_argument('--name', dest='project_name', default='protonet')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--vali', dest='val_iter', default=60, type=int)
    parser.add_argument('--frac', dest='gpu_frac', default=0.44, type=float)
    parser.add_argument('--config', dest='config', default='general')
    args = parser.parse_args()
    return args

def validate(test_net, test_gen):
    #np.random.seed(2)
    target_dataset = test_gen.dataset_list
    print ('Validation-')
    for tdi, dset in enumerate(target_dataset):
        accs, losses = [], []
        for _ in range(args.val_iter):
            sx, sy, qx, qy = test_gen.get_episode(args.nway, 
                    args.kshot, args.qsize, 
                    dataset_name=dset)
            fd = {\
                test_net.inputs['sx']: sx,
                test_net.inputs['qx']: qx,
                test_net.inputs['qy']: qy}
            outputs = [test_net.outputs['acc'], test_net.outputs['loss']]
            acc, loss = sess.run(outputs, fd)
            accs.append(acc)
            losses.append(loss)
        print ('{:12s} | ACC: {:.3f} ({:.3f})'
            '| LOSS: {:.3f}   '\
            .format(dset,
            np.mean(accs) * 100., 
            np.std(accs) * 100. * 1.96 / np.sqrt(args.val_iter),
            np.mean(losses)))
    #np.random.seed()

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
    test_kshot = args.kshot
    
    with tf.variable_scope(args.project_name):
        lr_ph = tf.placeholder(tf.float32) 
        protonet = ProtoNet(args.dataset_name, nway, kshot, qsize, isTr=True)
        loss = protonet.outputs['loss']
        acc = protonet.outputs['acc']

        # only evaluates 5way - kshot
        test_net = ProtoNet(args.dataset_name, nway,
                test_kshot, qsize, isTr=False, reuse=True)

        opt = tf.train.AdamOptimizer(lr_ph) 
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_op = opt.minimize(loss) 
        saver = tf.train.Saver()
        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    if args.pretrained:
        loc = os.path.join(args.model_dir,
                args.project_name,
                args.dataset_name + '.ckpt')
        saver.restore(sess, loc)
    
    config = load_config(args.config)
    train_gen = EpisodeGenerator(args.dataset_dir, 'train', config)
    #test_gen = EpisodeGenerator(args.dataset_dir, 'test', config)
    if args.train:
        max_iter = train_gen.dataset_size[args.dataset_name] * args.max_epoch \
                // (nway * qsize)
        show_step = args.show_epoch * max_iter // args.max_epoch
        save_step = args.save_epoch * max_iter // args.max_epoch
        avger = np.zeros([4])
        for i in range(1, max_iter+1): 
            stt = time.time()
            cur_epoch = i * (nway * qsize) // train_gen.dataset_size[args.dataset_name]
            lr = args.lr if i < 0.7 * max_iter else args.lr*.1
            sx, sy, qx, qy = train_gen.get_episode(nway, kshot, qsize, 
                    dataset_name=args.dataset_name)
            fd = {\
                protonet.inputs['sx']: sx,
                protonet.inputs['qx']: qx,
                protonet.inputs['qy']: qy,
                lr_ph: lr}
            p1, p2, _ = sess.run([acc, loss, train_op], fd)
            avger += [p1, p2, 0, time.time() - stt] 

            if i % show_step == 0 and i != 0: 
                avger /= show_step
                print ('========= {:15s} epoch : {:8d}/{} ========='\
                        .format(args.dataset_name, cur_epoch, args.max_epoch))
                print ('Training - ACC: {:.3f} '
                    '| LOSS: {:.3f}   '
                    '| lr : {:.3f}    '
                    '| in {:.2f} secs '\
                    .format(avger[0], 
                        avger[1], lr, avger[3]*show_step))
                #validate(test_net, test_gen)
                avger[:] = 0

            if i % save_step == 0 and i != 0: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.project_name, # bline/
                        args.dataset_name + '.ckpt')  # cifar100.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(test_net, test_gen)
