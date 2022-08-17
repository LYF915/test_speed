import torch
import os
import onnxruntime
import psutil
import time
from .utils import profile

def add_path():
    # Change to True when onnxruntime (like onnxruntime-gpu 1.0.0 ~ 1.1.2) cannot be imported.
    add_cuda_path = True

    # For Linux, see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
    # Below is example for Windows
    if add_cuda_path:
        cuda_dir = '/usr/local/cuda-11.1/bin'
        cudnn_dir = '/usr/local/cuda-11.1/bin'
        if not (os.path.exists(cuda_dir) and os.path.exists(cudnn_dir)):
            raise ValueError("Please specify correct path for CUDA and cuDNN. Otherwise onnxruntime cannot be imported.")
        else:
            if cuda_dir == cudnn_dir:
                os.environ["PATH"] = cuda_dir + ';' + os.environ["PATH"]
            else:
                os.environ["PATH"] = cuda_dir + ';' + cudnn_dir + ';' + os.environ["PATH"]

def export_onnx(inputs,model,export_model_path,opset_version=11):
    model.eval()
    if not os.path.exists(export_model_path):
        if not os.path.exists(os.path.dirname(export_model_path)):
            os.mkdir(os.path.dirname(export_model_path))

        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(model,                                            # model being run
                            args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                            f=export_model_path,                              # where to save the model (can be a file or file-like object)
                            opset_version=opset_version,                      # the ONNX version to export the model to
                            do_constant_folding=True,                         # whether to execute constant folding for optimization
                            input_names=['input_ids',                         # the model's input names
                                        'attention_mask', 
                                        'token_type_ids'],
                            output_names=['output'],           # the model's output names
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                            'attention_mask' : symbolic_names,
                                            'token_type_ids' : symbolic_names})
            print("Model exported at ", export_model_path)


@profile("onnx")
def inference_profile(inputs,session,num_runs=1000):
    with torch.no_grad():
        _ = session.run(None, inputs)
    return


def inference(inputs,session,num_runs=1000,batch_size=128):
    def run_model(inputs):
        with torch.no_grad():
            session.run(None,inputs)

    start = time.time()
    for _ in range(num_runs):
        run_model(inputs)
    torch.cuda.synchronize()
    total_time=time.time() - start
    print(f"total time: {total_time} ms")
    print(f"{num_runs/total_time * batch_size} Sentences / s")
    print(f"{total_time/num_runs/batch_size * 1000} ms / Sentences ")

    return


def onnx_infer_time_count_gpu(model,inputs,export_model_path,batch_size,num_runs=1000):
    add_path()

    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    assert onnxruntime.get_device() == 'GPU'

    device_name = 'gpu'
    sess_options = onnxruntime.SessionOptions()
    # print(export_model_path)
    
    dir_name=os.path.dirname(export_model_path)
    file_name=os.path.basename(export_model_path).split(".")[0]
    # Note that this will increase session creation time so enable it for debugging only.
    # sess_options.optimized_model_filepath = str(os.path.join(dir_name,f"{file_name}-optimized_model_{device_name}.onnx"))
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Please change the value according to best setting in Performance Test Tool result.
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
   
    session = onnxruntime.InferenceSession(export_model_path, sess_options,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    
    # io_binding = session.io_binding()
    # io_binding.bind_input(input)
    # # io_binding.bind_output(label_name, 'cuda')
    # data = ort.OrtValue.ortvalue_from_numpy(inputs, 'cuda', 0)
    # io_binding.bind_input(input_name, 'cuda', 0, np.float32, [8, 3, 480, 480], data.data_ptr())
    

    # warmup!!!
    print("warmup!")
    inference(inputs,session,100,batch_size)
    print("profile!")
    inference_profile(inputs,session,int(num_runs/100))
    print("test!")
    # start = time.time()
    inference(inputs,session,num_runs,batch_size)
    # total_time=time.time() - start

    
