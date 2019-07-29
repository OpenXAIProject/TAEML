# 5-way 1-shot miniImagenet test
# reported result : 48.70 (1.84) 
# reproduced result : 48.564 (0.840)
# - reported result is acheived from experimets with query size 1 (--qs 1)
# and this is not a convention for the ordinary few-shot classification test,
# which is the query size is 15. The size of test set is 15 times smaller
# than ordinary one, and that is why the variance is that much high rather than
# other methods (ex. Protonet - 49.42 (0.78), RelationNet - 50.44 (0.82))
# Also, batch normalization is always set to train mode, and it is little 
# problematic because the query size is dependent on the performance 
# at the test time.


gpu=6
K=1  # kshot
MtLr=1e-3  # outer gradient descent step size
InLr=3e-2  # inner gradient descent step size
STG=0     # stop gradient
InIt=5    # inner loop iteration

params=K${K}_MtLr${MtLr}_STG${STG}_InIt${InIt}_InLr${InLr}
# if you want test, uncomment resume and the last line
#resume=models/mamlnet/${params}_30000

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --ks $K \
    --stop_grad ${STG} \
    --mt_lr ${MtLr} \
    --in_lr ${InLr} \
    --ini  ${InIt} \
    --parm ${params} \
    #--train 0 --resume ${resume} --vali 600 --qs 15 \
