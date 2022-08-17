import random
import torch
import numpy as np
import jax.numpy as jnp
import jax
import functools
import os

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def data_process(args,batch_size,length):
    
    if not args.no_cuda:
        batch = {
            "input_ids": torch.ones([batch_size, length], dtype=torch.int32).cuda(),
            "attention_mask": torch.ones([batch_size, length], dtype=torch.int32).cuda(),
            "token_type_ids": torch.ones([batch_size, length], dtype=torch.int32).cuda(),
        }  
    else:
        batch = {
            "input_ids": torch.ones([batch_size, length], dtype=torch.int32).cpu(),
            "attention_mask": torch.ones([batch_size, length], dtype=torch.int32).cpu(),  
            "token_type_ids": torch.ones([batch_size, length], dtype=torch.int32).cpu(),
        }  

    inputs = {"input_ids": batch["input_ids"],
              "attention_mask": batch["attention_mask"]}
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch["token_type_ids"] if args.model_type in [
                "bert",
                "masked_bert",
                "xlnet",
                "albert"] else None)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    return inputs


def data_process_onnx(args,batch_size,length):
    
    batch = {
        "input_ids": torch.ones([batch_size, length], dtype=torch.int32).detach().cpu().numpy(),
        "attention_mask": torch.ones([batch_size, length], dtype=torch.int32).detach().cpu().numpy(),
        "token_type_ids": torch.ones([batch_size, length], dtype=torch.int32).detach().cpu().numpy(),
    }  

    inputs = {"input_ids": batch["input_ids"],
              "attention_mask": batch["attention_mask"]}
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch["token_type_ids"] if args.model_type in [
                "bert",
                "masked_bert",
                "xlnet",
                "albert"] else None)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    return inputs

def data_process_jax(args,batch_size,length):
    
    batch = {
        "attention_mask": jax.random.randint(key=jax.random.PRNGKey(42), shape=[batch_size, length], dtype="int32", minval=0, maxval=99999),
        "input_ids": jax.random.randint(key=jax.random.PRNGKey(42), shape=[batch_size, length], dtype="int32", minval=0, maxval=99999),
        "token_type_ids": jax.random.randint(key=jax.random.PRNGKey(42), shape=[batch_size, length], dtype="int32", minval=0, maxval=99999),
    }  

    inputs = {"input_ids": batch["input_ids"],
              "attention_mask": batch["attention_mask"]}
              
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch["token_type_ids"] if args.model_type in [
                "bert",
                "masked_bert",
                "xlnet",
                "albert"] else None)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
    # # print(inputs)
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")
    return inputs


def device_usage(device='cuda'):

    # print('Using device:', device)
    # print()

    # print('GPU Device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
    # print()

    #Additional Info when using cuda
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


def profile(method_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args,**kw):
            N = args[-1]
            bsz=os.environ["bsz"]
            file_path=f"/cephfs/luoyifei/work/test_speed/output/output3_{bsz}/{method_type}"

            if os.path.exists(file_path):
                import shutil
                shutil.rmtree(file_path,ignore_errors=True)

            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU,
                    ],
                    # In this example with wait=1, warmup=3, active=7,
                    # profiler will skip the first step/iteration,
                    # start warming up on the second, record
                    # the third and the forth iterations,
                    # after which the trace will become available
                    # and on_trace_ready (when set) is called;
                    # the cycle repeats starting with the next step
                    schedule=torch.profiler.schedule(wait=3,
                                                    warmup=3,
                                                    active=14,),
                    # on_trace_ready=trace_handler,
                    profile_memory=True,
                    record_shapes=True,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(file_path),
                    # used when outputting for tensorboard
            ) as p:
                # N = args.num_runs
                for i in range(N):
                    func(*args, **kw)
                    if i == 4 : device_usage()
                    p.step()
            return 
        return wrapper
    return decorator


