import os
import multiprocessing

print(int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count())))
