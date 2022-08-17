import argparse
import torch
import os
from transformers import BertForMaskedLM,FlaxBertForMaskedLM
from transformers import FlaxBertForMaskedLM
from speed.utils import set_seed,data_process,data_process_onnx,data_process_jax
from speed import torch_infer_time_count_gpu,jax_infer_time_count_gpu,onnx_infer_time_count_gpu,export_onnx

model_class={
    "torch":BertForMaskedLM,
    "onnx":BertForMaskedLM,
    "jax":FlaxBertForMaskedLM,
}

# https://github.com/microsoft/onnxruntime/issues/10789


global max_seq_length
max_seq_length=128

def main():
    parser =argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",default=None,type=str)
    parser.add_argument("--model_type",default="bert",type=str)
    parser.add_argument("--type",default=None,type=str)
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=None,
        type=int,
        help="Batch size per GPU/CPU for evaluating.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--n_gpu", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_runs", type=int, default=2000, help="For distributed training: local_rank")

    args = parser.parse_args()
    set_seed(args)

    print(f"*****{args.type} Speed ******")
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    # world_size = int(os.getenv('WORLD_SIZE', '1'))

    args.device = device
    
    batch_size = args.per_gpu_eval_batch_size

    print(f"args.type:{args.type}")
    model = model_class[args.type].from_pretrained(args.model_name_or_path) 

    if args.type!="jax":
        print(f"model moving to {args.device}")
        model.to(args.device)

    if args.type== "torch":
        inputs = data_process(args,batch_size=batch_size,length=max_seq_length)
    elif args.type=="onnx":
        inputs = data_process(args,batch_size=batch_size,length=max_seq_length)
        export_model_path="./onnx/"
        export_model_path=os.path.join(export_model_path,f"{batch_size}-{max_seq_length}-bert.onnx")
        export_onnx(inputs,model,export_model_path,opset_version=11)

        inputs = data_process_onnx(args,batch_size=batch_size,length=max_seq_length)
    elif args.type=="jax":
        inputs = data_process_jax(args,batch_size=batch_size,length=max_seq_length)
    else:
        raise ValueError(f"not implement {args.type}")


    if args.type=="torch":
        model.eval()
        torch_infer_time_count_gpu(model,inputs,batch_size=batch_size,num_runs=args.num_runs)
    elif args.type=="jax":
        jax_infer_time_count_gpu(model,inputs,batch_size=batch_size,num_runs=args.num_runs)
    elif args.type=="onnx":
        # onnx_infer_time_count_gpu(model,inputs,batch_size=batch_size,num_runs=1000)
        onnx_infer_time_count_gpu(model,inputs,export_model_path,batch_size=batch_size,num_runs=args.num_runs)
    
if __name__ == "__main__":
    main()


# def torch_infer_time_count_gpu(model,inputs,batch_size,num_runs=1000):
#     model.eval()
#     # warmup!!!
#     for i in range(1000):
#         with torch.no_grad():
#             outputs = model(**inputs)

#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     torch.cuda.synchronize()
#     start.record()
#     for i in range(num_runs):
#         with torch.no_grad():
#             outputs = model(**inputs)
#     end.record()
#     torch.cuda.synchronize()
#     total_time = start.elapsed_time(end) / 1000  # s

#     print(f"{num_runs/total_time * batch_size} Sentences / s")
#     print(f"{total_time/num_runs/batch_size * 1000} ms / Sentences ")





    