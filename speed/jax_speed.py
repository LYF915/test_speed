import jax
import torch
import time
from .utils import profile

@profile("jax")
def inference_profile(inputs,model,num_runs=1000):
    model(**inputs)
    return

def inference(inputs,model,num_runs=1000,batch_size=128):
    @jax.jit
    def run_model(**inputs):
        _ = model(**inputs)
        return

    start = time.time()
    for _ in range(num_runs):
        run_model(**inputs)
    torch.cuda.synchronize()
    total_time=time.time() - start
    print(f"total time: {total_time} ms")
    print(f"{num_runs/total_time * batch_size} Sentences / s")
    print(f"{total_time/num_runs/batch_size * 1000} ms / Sentences ")
    return

def jax_infer_time_count_gpu(model,inputs,batch_size,num_runs=1000):
    # model.eval()
    # warmup!!!
    print("warmup!")
    inference(inputs,model,100,batch_size)

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # torch.cuda.synchronize()
    # start.record()
    # inference(inputs,model,num_runs)
    # end.record()
    # torch.cuda.synchronize()
    # total_time = start.elapsed_time(end) / 1000  # s
    print("profile!")
    inference_profile(inputs,model,int(num_runs/100))

    print("test!")
    # start = time.time()
    inference(inputs,model,num_runs,batch_size)
    # total_time=time.time() - start
    # print(f"total time: {total_time} ms")
    # print(f"{num_runs/total_time * batch_size} Sentences / s")
    # print(f"{total_time/num_runs/batch_size * 1000} ms / Sentences ")