#!/bin/sh
#SBATCH -p JX-GPU-IB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH -o %j-%N-%2t.log
#SBATCH --mem 256g
#SBATCH -J speed


# export TORCH_EXTENSIONS_DIR=./tmp
# export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"

# echo $LD_LIBRARY_PATH

export LD_LIBRARY_PATH="/nfs/users/luoyifei/anaconda3/envs/intel_compress/lib/"

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64/

export bsz=512

kill -9 $(ps aux | grep python | grep -v grep | awk '{print $2}')
python -u test_speed.py \
    --model_name_or_path bert-base-uncased \
    --per_gpu_eval_batch_size $bsz \
    --type torch


kill -9 $(ps aux | grep python | grep -v grep | awk '{print $2}')
python -u test_speed.py \
    --model_name_or_path bert-base-uncased \
    --per_gpu_eval_batch_size $bsz \
    --type jax 



kill -9 $(ps aux | grep python | grep -v grep | awk '{print $2}')
python -u test_speed.py \
    --model_name_or_path bert-base-uncased \
    --per_gpu_eval_batch_size $bsz \
    --type onnx


