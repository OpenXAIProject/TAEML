gpu=1
N=10
K=5
Q=3
ex_type=tiered_test
name=Proto_${ex_type}_${N}n_${K}k # should be matched with the name in run_baselearner.sh

# training taeml model
CUDA_VISIBLE_DEVICES=$gpu python taeml.py \
    --name $name \
    --ks $K \
    --nw $N \
    --qs $Q \
    --maxe 10000 \
    --config $ex_type \
    --trd # if non-transductive mode: comment this line

# test it
# query size can be changed in (3, 15)
# which means BQ=q*n with transductive setting only
CUDA_VISIBLE_DEVICES=$gpu python get_results.py \
    --name $name \
    --qs $Q \
    --ks $K \
    --nw $N \
    --config $ex_type \
    --trd
