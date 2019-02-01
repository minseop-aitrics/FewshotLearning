gpu=5
W=5 # n-way 
S=1 # k-shot
data="tieredImagenet"
name=${W}w${S}s_protonet_${data}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

# to get test result, uncomment last line
#echo $log
#exec &> >(tee -a "$log")
CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
    --nw $W --ks $S --name $name --data ../data_npy/${data} \
    #--pr 1 --train 0 --vali 600


# ------------------
# results report
# ------------------
# trained on 5way-1shot 
# tested on 5way-1shot (matched way-shot)
# Acc : 52.547 (0.766)
# maybe its because of the fused batch-norm ??
