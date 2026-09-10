import nvidia_smi
import torch
import logging


def print_gpu_usage():
    # nvidia_smi.nvmlInit()

    # deviceCount = nvidia_smi.nvmlDeviceGetCount()
    # for i in range(deviceCount):
    #     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    #     info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    #     print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(
            f"|Device {i}| Mem Free: {mem.free/1024**3:5.2f}GB / {mem.total/1024**3:5.2f}GB | gpu-util: {util.gpu:.1f} % | gpu-mem: {util.memory:3.1f}% |"
        )
    nvidia_smi.nvmlShutdown()


def log_gpu_info():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated("cuda") / (1024**3)
        reserved = torch.cuda.memory_reserved("cuda") / (1024**3)
        max_allocated_memory = torch.cuda.max_memory_allocated("cuda") / (1024**3)
        max_reserved_memory = torch.cuda.max_memory_reserved("cuda") / (1024**3)

        logging.info(f"Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total Visible GPU Memory: [{total/(1024**3):.2f}] GB")
        logging.info(f"Free GPU Memory: [{free/(1024**3):.2f}] GB")
        logging.info(f"Visible GPUs count: [{torch.cuda.device_count()}] GB")
        logging.info(f"Memory allocated: [{allocated:.2f}] GB")
        logging.info(f"Memory reserved: [{reserved:.2f}] GB")
        logging.info(f"Max memory allocated: [{max_allocated_memory:.2f}] GB")
        logging.info(f"Max memory reserved: [{max_reserved_memory:.2f}] GB")
    else:
        logging.info("GPU is not available")
