gpu=3
W=5
S=1
#name=${W}w${N}n${S}s_finset_5con
name=${W}w${S}s_protonet_baseline
log="models/${name}/log.txt"
mkdir -p models/${name}

#echo $log
#exec &> >(tee -a "$log")
CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
    --nw $W --ks $S --name $name --data ../miniImagenet \
    --pr 1 --train 0


# ------------------
# results report
# ------------------
# trained on 5way-1shot 
# tested on 5way-1shot (matched way-shot)
# Acc : 51.780 (0.973) 
# maybe its because of the fused batch-norm ??
