#!/bin/sh

cd ..
mkdir -p raw

mkdir -p raw/mnist
cd raw/mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
cd ../../

mkdir -p raw/cifar10
cd raw/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xf cifar-10-python.tar.gz
cd ../../

mkdir -p raw/cifar100
cd raw/cifar100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xf cifar-100-python.tar.gz
cd ../../

mkdir -p raw/caltech101
cd raw/caltech101
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xf 101_ObjectCategories.tar.gz
cd ../../

mkdir -p raw/caltech256
cd raw/caltech256
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar
cd ../../

mkdir -p raw/voc2012
cd raw/voc2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
cd ../../

mkdir -p raw/cub200_2011
cd raw/cub200_2011
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xf CUB_200_2011.tgz
cd ../../

mkdir -p raw/awa2
cd raw/awa2
wget https://cvml.ist.ac.at/AwA2/AwA2-data.zip
unzip AwA2-data.zip -d ./
cd ../../

mkdir -p raw/omniglot
cd raw/omniglot
wget "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip"
wget "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip"
unzip -q '*.zip'
cd ../../

mkdir -p raw/miniImagenet
cd raw/miniImagenet
g1=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY
g2=mini-imagenet.tar.gz
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$g1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$g1" -O $g2
rm -rf /tmp/cookies.txt
tar -xf mini-imagenet.tar.gz
cd ../../

mkdir -p raw/tieredImagenet
cd raw/tieredImagenet
g1=1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u
g2=tiered-imagenet.tar
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$g1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$g1" -O $g2
rm -rf /tmp/cookies.txt
tar -xvf tiered-imagenet.tar
cd ../../


cd dataset-serializer

# generate datasets
python read_datasets.py
python pkl2dataset.py
