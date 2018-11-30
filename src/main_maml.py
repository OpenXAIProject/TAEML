import tensorflow as tf
import numpy as np
import argparse
import time
import os
from lib.episode_generator import EpisodeGenerator
from lib.maml import MAML
from lib.params import TRAIN_DATASETS_SIZE, TRAIN_DATASETS, TEST_DATASETS, TEST_NUM_EPISODES

from queue import Queue
from threading import Thread
from functools import reduce

def ep_generator(q, meta_batch_size, nway, kshot, qsize):
    while True:
        inputa, inputb, labela, labelb = [], [], [], []
        for _ in range(meta_batch_size):
            inputa_, labela_, _, _ = ep_gen.get_episode(nway, kshot, qsize, dataset_name='cifar100')
            inputb_, labelb_, _, _ = ep_gen.get_episode(nway, kshot, qsize, dataset_name='cifar100')
            inputa.append(inputa_)
            inputb.append(inputb_)
            labela.append(labela_)
            labelb.append(labelb_)
        inputa = np.stack(inputa, axis=0)
        inputb = np.stack(inputb, axis=0)
        labela = np.stack(labela, axis=0)
        labelb = np.stack(labelb, axis=0)
        q.put((inputa, labela, inputb, labelb))

parser = argparse.ArgumentParser(description='normalized protonet')
parser.add_argument('--init', dest='initial_step', default=0, type=int)
parser.add_argument('--maxe', dest='max_epoch', default=100, type=int)
parser.add_argument('--meta_batch_size', default=2, type=int)
parser.add_argument('--num_updates', default=5, type=int)
parser.add_argument('--test_num_updates', default=10, type=int)

parser.add_argument('--update_lr', default=1e-2, type=float)
parser.add_argument('--meta_lr', default=1e-3, type=float)

parser.add_argument('--qs', dest='qsize', default=15, type=int)
parser.add_argument('--nw', dest='nway', default=5, type=int)
parser.add_argument('--ks', dest='kshot', default=1, type=int)
parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
parser.add_argument('--data', dest='dataset_dir', default='datasets')
parser.add_argument('--model', dest='model_dir', default='../models')
parser.add_argument('--name', dest='model_name', default='maml_baseline')
parser.add_argument('--gpu_num', type=int, default=0)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

print ('='*50)
print ('args::')
for arg in vars(args):
    print ('%15s: %s'%(arg, getattr(args, arg)))
print ('='*50)

ep_queue = Queue(20)

nway = args.nway
kshot = args.kshot
qsize = args.qsize
meta_batch_size = args.meta_batch_size
num_updates = args.num_updates
test_num_updates = args.test_num_updates

meta_lr = args.meta_lr
update_lr = args.update_lr

inputa = tf.placeholder(tf.float32, [meta_batch_size, nway*kshot,84,84,3])
inputb = tf.placeholder(tf.float32, [meta_batch_size, nway*kshot,84,84,3])
labela = tf.placeholder(tf.float32, [meta_batch_size, nway*kshot,nway])
labelb = tf.placeholder(tf.float32, [meta_batch_size, nway*kshot,nway])
is_train = tf.placeholder(tf.bool, [])

config = {}
config['update_lr'] = update_lr
config['test_num_updates'] = test_num_updates
config['num_updates'] = num_updates
config['is_train'] = is_train
config['num_classes'] = nway

sess = tf.Session()
with tf.variable_scope(args.model_name):
    model = MAML(config)
    loss, acc, output = model.train(inputa, inputb, labela, labelb)

opt = tf.train.AdamOptimizer(meta_lr)
#update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_op):
gvs = opt.compute_gradients(loss)
gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
train_op = opt.apply_gradients(gvs)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

ep_gen = EpisodeGenerator(args.dataset_dir, 'train')
worker = Thread(target=ep_generator, args=(ep_queue,
                                           meta_batch_size, nway, kshot, qsize))
worker.setDaemon(True)
worker.start()

datasets_size = reduce((lambda x,y:x+y),
                        [TRAIN_DATASETS_SIZE[key] for key in TRAIN_DATASETS])
max_iter = datasets_size * args.max_epoch // (nway * qsize)

show_step = args.show_epoch * max_iter // args.max_epoch
save_step = args.save_epoch * max_iter // args.max_epoch
avger = np.zeros([4])

for i in range(1, max_iter+1):
    stt = time.time()
    cur_epoch = i * (nway * qsize) // datasets_size
    #lr = args.lr if i < 0.7 * max_iter else args.lr*.1

    inputa_, labela_, inputb_, labelb_ = ep_queue.get()
    fd = {inputa: inputa_, labela: labela_,
          inputb: inputb_, labelb: labelb_, is_train: True}
    p1, p2, _ = sess.run([acc, loss, train_op], fd)
    avger += [p1, p2, 0, time.time() - stt]

    if i % show_step == 0 and i != 0:
        avger /= show_step
        print('epoch : {:8d}/{}  ACC: {:.3f}   |   LOSS: {:.3f}   |  lr : {:.3f}   | in {:.2f} secs'\
                .format(cur_epoch, args.max_epoch, avger[0],
                    avger[1], meta_lr, avger[3]*show_step))
        print(sess.run(output, fd)[0][0])
        avger[:] = 0

    '''
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
    '''
