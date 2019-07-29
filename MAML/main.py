import tensorflow as tf 
import numpy as np 
import argparse
import time
import os
import pdb
from lib.episode_generator import EpisodeGenerator
from lib.networks import MAMLNet

def parse_args():
    parser = argparse.ArgumentParser(description='normalized protonet')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxi', dest='max_iter', default=30000, type=int)
    parser.add_argument('--qs', dest='qsize', default=15, type=int)
    parser.add_argument('--nw', dest='nway', default=5, type=int)
    parser.add_argument('--ks', dest='kshot', default=1, type=int)
    parser.add_argument('--showi', dest='show_step', default=100, type=int)
    parser.add_argument('--savei', dest='save_step', default=5000, type=int)
    parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
    parser.add_argument('--data', dest='dataset_dir', default='../miniImagenet')
    parser.add_argument('--model', dest='model_dir', default='models')
    parser.add_argument('--dset', dest='dataset_name', default='miniImagenet')
    parser.add_argument('--name', dest='model_name', default='mamlnet')
    parser.add_argument('--parm', dest='param_str', default='default')
    parser.add_argument('--gpufrac', dest='gpufraction', default=0.90, type=float)
    parser.add_argument('--in_lr', dest='inner_lr', default=1e-2, type=float)
    parser.add_argument('--mt_lr', dest='meta_lr', default=1e-3, type=float)
    parser.add_argument('--vali', dest='val_iter', default=50, type=int)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--resume', dest='resume', default=None)
    parser.add_argument('--stop_grad', dest='stop_grad', default=0, type=int)
    parser.add_argument('--ini', dest='inner_loop_iter', 
            default=5, type=int)
    args = parser.parse_args()
    return args

def validate(test_net, ep_test):
    val_losses, val_accs = [], []
    np.random.seed(2)
    for _ in range(args.val_iter):
        sx, sy, qx, qy = [], [], [], []
        for _ in range(1): # only one meta-batch
            sxt, syt, qxt, qyt = ep_test.get_episode(nway, kshot, qsize)
            sx.append(sxt)
            sy.append(syt)
            qx.append(qxt)
            qy.append(qyt)
        ip = test_net.inputs
        feed_dict = {\
                ip['sx']: sx, ip['sy']: sy,
                ip['qx']: qx, ip['qy']: qy,
                ip['lr_alpha']: args.inner_lr}
        op = test_net.outputs
        run_list = [op['accuracy'], op['lossb']]
        
        acc, loss_b = sess.run(run_list, feed_dict)
        val_losses.append(np.mean(loss_b))
        val_accs.append(np.mean(acc))

    print ('Validation - ACC: {:.3f} ({:.3f})'
        '| LOSS: {:.3f}   '\
        .format(np.mean(val_accs) * 100., 
        np.std(val_accs) * 100. * 1.96 / np.sqrt(args.val_iter),
        np.mean(loss_b)))
    np.random.seed()

if __name__=='__main__': 
    args = parse_args() 
    print ('='*50) 
    print ('args::') 
    if args.kshot == 1:
        args.meta_batch_size = 4
    if args.kshot == 5:
        args.meta_batch_size = 2
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    nway = args.nway
    kshot = args.kshot
    qsize = args.qsize 
    
    if args.train:
        mamlnet = MAMLNet(args.model_name, nway, kshot, qsize, 
            args.meta_batch_size,
            inner_loop_iter=args.inner_loop_iter, 
            stop_grad=args.stop_grad)
        # build trainnet only when is needed

    test_net = MAMLNet(args.model_name, nway, kshot, qsize, mbsize=1,
            reuse=True if args.train else False, inner_loop_iter=10, isTr=False)
        
    sess = tf.Session() 
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()

    if args.resume is not None:
        print ('restore from at : {}'.format(args.resume))
        saver.restore(sess, args.resume)

    ep_train = EpisodeGenerator(args.dataset_dir, 'train')
    ep_test = EpisodeGenerator(args.dataset_dir, 'test')

    vlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i, v in enumerate(vlist):
        print (i, v)

    if args.train:
        avger = np.zeros([4])
        for i in range(1, args.max_iter+1):
            stt = time.time()
            lr = args.meta_lr if i < 0.7 * args.max_iter else args.meta_lr*.1

            sx = []; sy = []; qx = []; qy = []
            for _ in range(args.meta_batch_size):
                sxt, syt, qxt, qyt = ep_train.get_episode(nway, kshot, qsize)
                sx.append(sxt)
                sy.append(syt) 
                qx.append(qxt) 
                qy.append(qyt)

            # qx : (metabatchsize, nk, img_size)
            # sx : (metabatchsize, nq, img_size)
            ip = mamlnet.inputs
            feed_dict = {ip['sx']: sx, ip['sy']: sy, 
                    ip['qx']: qx, ip['qy']: qy,
                    ip['lr_alpha']: args.inner_lr, ip['lr_beta']: lr}
            op = mamlnet.outputs
            run_list = [op['accuracy'], op['lossb'], mamlnet.train_op]

            
            acc, loss_b, _ = sess.run(run_list, feed_dict)
            avger += [np.mean(acc), np.mean(loss_b[-1]), 0, time.time() - stt]
            
#            p1, p2 = sess.run([mamlnet.outputs['lvlossb'],
#                mamlnet.outputs['lossb']], feed_dict)
#            print (np.mean(p1, (1,2)))
#            print (np.mean(p2, (1,2)))
#            
#            pdb.set_trace()

            # get validation stats
            if i % args.show_step == 0 and i != 0: 
                avger /= args.show_step
                print ('========= STEP : {:8d}/{} ========='\
                        .format(i, args.max_iter))
                print ('Training - ACC: {:.3f} '
                    '| LOSS: {:.3f}   '
                    '| lr : {:.3f}    '
                    '| in {:.2f} secs '\
                    .format(avger[0], 
                        avger[1], lr, avger[3]*args.show_step))
                avger[:] = 0
                validate(test_net, ep_test)



            if i % args.save_step == 0 and i != 0: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.model_name, # mamlnet/
                        args.param_str) + '_{}'.format(i) # 5way_1shot_model.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(test_net, ep_test)
