# 5-way 5-shot miniImagenet test
# reported result : 63.11 (0.91) 
# reproduced result : 64.127 (0.721)


gpu=7
K=5  # kshot
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
