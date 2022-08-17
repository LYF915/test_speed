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


export bsz=468
kill -9 $(ps aux | grep python | grep -v grep | awk '{print $2}')
python -u test_speed.py \
    --model_name_or_path bert-base-uncased \
    --per_gpu_eval_batch_size $bsz \
    --type onnx


