N=10 # n-way
K=5 # k-shot
ex_type=general # see config/loader.py
name=Proto_${ex_type}_${N}n_${K}k
mkdir -p ../models/${name}

function srun(){
    CUDA_VISIBLE_DEVICES=$1 python train_baselearners.py \
        --name $2 --nw $3 --ks $3 --config $5 --maxe $6 --frac $7 --dset $8
}

# for tiered imageenet setting (when ex_type=tiered or tiered_test)
#srun 4 ${name} $N $K $ex_type 40 0.41 tiered_sub0 &
#srun 4 ${name} $N $K $ex_type 40 0.41 tiered_sub1 &
#srun 5 ${name} $N $K $ex_type 40 0.41 tiered_sub2 &
#srun 5 ${name} $N $K $ex_type 40 0.41 tiered_sub3 &

# for general experiment setting (ex_type=general) which generates result table in the paper
srun 4 ${name} $N $K $ex_type 100 0.41 awa2 &
srun 4 ${name} $N $K $ex_type 100 0.41 cifar100 &
srun 5 ${name} $N $K $ex_type 100 0.41 omniglot &
srun 5 ${name} $N $K $ex_type 100 0.41 voc2012 &
srun 6 ${name} $N $K $ex_type 100 0.41 caltech256 &
srun 6 ${name} $N $K $ex_type 100 0.41 multiple 

# for test: run following line
# CUDA_VISIBLE_DEVICES=7 python test_baselearners.py --name $name --ks $K --nw $N --config $ex_type
