#/bin/env/bash
import os
import subprocess
import multiprocessing

txt_path = "training_data/nba_dataset/data_rest_pose.txt"

def work(cmd):
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# cmds = []
# with open(txt_path, "r") as fp:
#     for line in fp.readlines():
#         line = line.strip("\n")
#         cmd = (
#             "python -m apps.render_data -i "
#             f"{line} "
#             "-o training_data/nba_dataset/rest_pose -t nba -e"
#         )
#         cmds.append(cmd)

txt_path = "training_data/nba_dataset/data_2ku.txt"
cmds = []
with open(txt_path, "r") as fp:
    for line in fp.readlines():
        line = line.strip("\n")
        cmd = (
            "python -m apps.render_data -i "
            f"{line} "
            "-o training_data/nba_dataset/2ku -t nba -e"
        )
        cmds.append(cmd)

count = multiprocessing.cpu_count()
pool = multiprocessing.Pool(count-1)
pool.map(work, cmds)
