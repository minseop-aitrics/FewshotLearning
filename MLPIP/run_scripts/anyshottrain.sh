# 5-way 1-shot miniImagenet test
# reported result : 53.40 ± 1.82
# reproduced result : 

gpu=2
K=0  # kshot
#Lr=2.5e-4  # outer gradient descent step size
Lr=1e-3

params=K${K}_LR${Lr}_MLPIP
# if you want test, uncomment resume and the last line
#resume=models/mamlnet/${params}_30000

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --ks $K \
    --lr ${Lr} \
    --parm ${params} \
    #--train 0 --resume ${resume} --vali 600 --qs 15 \
