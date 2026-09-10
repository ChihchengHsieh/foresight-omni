import os, psutil
import logging

def print_mem(tag=""):
    p = psutil.Process(os.getpid())
    logging.info(f"[MEM] {tag}: {p.memory_info().rss / 1024**3:.2f} GB")

print_mem("after loading df")
print_mem("after labels")