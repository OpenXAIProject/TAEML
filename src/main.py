import tensorflow as tf
import numpy as np
import argparse
import time
import os
from lib.episode_generator import EpisodeGenerator
from lib.networks import ProtoNet
from utils.utils import cross_entropy, tf_acc
from lib.params import TRAIN_DATASETS_SIZE, TEST_DATASETS, TEST_NUM_EPISODES

from queue import Queue
from threading import Thread
from functools import reduce

def ep_generator(q, nway, kshot, qsize):
    while True:
        sx, sy, qx, qy = ep_gen.get_episode(nway, kshot, qsize)
        q.put((sx, sy, qx, qy))

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

parser = argparse.ArgumentParser(description='normalized protonet')
parser.add_argument('--init', dest='initial_step', default=0, type=int)
parser.add_argument('--maxe', dest='max_epoch', default=100, type=int)
parser.add_argument('--qs', dest='qsize', default=15, type=int)
parser.add_argument('--nw', dest='nway', default=5, type=int)
parser.add_argument('--ks', dest='kshot', default=1, type=int)
parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
parser.add_argument('--data', dest='dataset_dir', default='/home/sam/datasets/few-shot-tasks')
parser.add_argument('--model', dest='model_dir', default='../models')
parser.add_argument('--name', dest='model_name', default='verif_nb1s')
parser.add_argument('--gpufrac', dest='gpufraction', default=0.95, type=float)
parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
parser.add_argument('--norm', dest='norm', default=0, type=int,
        help="normalized prototypical network")
args = parser.parse_args()

print ('='*50)
print ('args::')
for arg in vars(args):
    print ('%15s: %s'%(arg, getattr(args, arg)))
print ('='*50)

ep_queue = Queue(20)

nway = args.nway
kshot = args.kshot
qsize = args.qsize

sx_ph = tf.placeholder(tf.float32, [nway*kshot,3,84,84])
qx_ph = tf.placeholder(tf.float32, [nway*qsize,3,84,84])
qy_ph = tf.placeholder(tf.float32, [nway*qsize,nway])
lr_ph = tf.placeholder(tf.float32)
isTr  = tf.placeholder(tf.bool, [])

with tf.variable_scope(args.model_name):
    model = ProtoNet()
    pred_dict = model.classifier(sx_ph, qx_ph, nway, isTr,
                                 trainable=True, normalized=args.norm)
    pred = pred_dict['pred']
    loss = cross_entropy(pred, qy_ph)
    acc = tf_acc(pred, qy_ph)

opt = tf.train.AdamOptimizer(lr_ph)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_op):
    train_op = opt.minimize(loss)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpufraction
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
if args.pretrained:
    model.restore(saver, sess)

ep_gen = EpisodeGenerator(args.dataset_dir, 'train')
worker = Thread(target=ep_generator, args=(ep_queue, nway, kshot, qsize))
worker.setDaemon(True)
worker.start()

ep_gen_test = EpisodeGenerator(args.dataset_dir, 'test')

datasets_size = reduce((lambda x,y:x+y), [val for key, val in TRAIN_DATASETS_SIZE.items()])
max_iter = datasets_size * args.max_epoch // (nway * qsize)
print(max_iter)
show_step = args.show_epoch * max_iter // args.max_epoch
save_step = args.save_epoch * max_iter // args.max_epoch
avger = np.zeros([4])

for i in range(1, max_iter+1):
    stt = time.time()
    cur_epoch = i * (nway * qsize) // datasets_size
    lr = args.lr if i < 0.7 * max_iter else args.lr*.1
    sx, sy, qx, qy = ep_queue.get()
    fd = {sx_ph: sx, qx_ph: qx, qy_ph: qy, lr_ph: lr, isTr: True}
    p1, p2, _ = sess.run([acc, loss, train_op], fd)
    avger += [p1, p2, 0, time.time() - stt]

    if i % show_step == 0 and i != 0:
        avger /= show_step
        print('epoch : {:8d}/{}  ACC: {:.3f}   |   LOSS: {:.3f}   |  lr : {:.3f}   | in {:.2f} secs'\
                .format(cur_epoch, args.max_epoch, avger[0],
                    avger[1], lr, avger[3]*show_step))
        avger[:] = 0

    if i % save_step == 0 and i != 0:
        out_loc = os.path.join(args.model_dir, # models/
                args.model_name, # bline/
                'all_datasets.ckpt')  # cifar100.ckpt
        model.save(saver, sess, out_loc)

        for test_set_name in TEST_DATASETS:
            test_accs = []
            for j in range(TEST_NUM_EPISODES):
                sx_t, sy_t, qx_t, qy_t = ep_gen_test.get_episode(nway, kshot, qsize,
                                                test_set_name)
                fd = {sx_ph: sx_t, qx_ph: qx_t, qy_ph: qy_t, isTr: False}
                acc_ = sess.run(acc, fd)
                test_accs.append(acc_)
            print('datset name: {}, test_acc: {:.3f} {:.5f}'.format(test_set_name,
                        np.mean(test_accs), np.std(test_accs)))
