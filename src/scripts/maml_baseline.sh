#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

python main_maml.py --ks 5 --nw 10 --name maml_baseline --maxe 100 --gpu_num $1
