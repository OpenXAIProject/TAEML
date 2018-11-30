g0=5
g1=6
g2=7

CUDA_VISIBLE_DEVICES=$g0 python main.py --ks 5 --nw 10 --name protonet_baseline --gpufrac 0.41 --maxe 100
