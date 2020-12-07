#!/usr/bin/env bash
"""
 This scripts loops through the directory and process the data to training data.
 (i) prt_util to calculate PRT
 (ii) render_data to render the data to uv mapping and 360 degrees mesh
"""

import os
import argparse
import multiprocessing
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser()

"""
e.g.
    -i data/nba_dataset/mesh/release, data/rp_dataset, data/twindom_dataset
    -o training_data/nba_dataset, training_data/rp_dataset etc.
"""
parser.add_argument("-i", "--input_dir", type=str, help="input directory")
parser.add_argument("-o", "--output_dir", type=str, help="training data directory")
parser.add_argument("-t", "--type", type=str, help="type of dataset")

args = parser.parse_args()

if not args.input_dir:
    raise ValueError("Missing input directory.")
if not args.output_dir:
    raise ValueError("Missing training data directory.")
if not args.type:
    raise ValueError("Missing dataset type.")

input_dir = args.input_dir
output_dir = args.output_dir
type = args.type

os.makedirs(output_dir, exist_ok=True)

def work(cmd):
    return subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

if type == "nba":
    player_2ku_dirs = []
    player_rest_dirs = []
    players_dir = os.listdir(input_dir)

    # separate them to two different array
    for p in players_dir:
        player_dir = os.path.join(input_dir, p)
        for style in ["rest_pose", "2ku"]:
            player_style_dir = os.path.join(player_dir, style)
            frame_dir = ""
            for f_dir in os.listdir(player_style_dir):
                if "NBA" in f_dir:
                    frame_dir = f_dir
                else:
                    continue
                player_mesh_dir = os.path.join(player_style_dir, frame_dir, "players")
                if style == "rest_pose":
                    player_rest_dirs.append(player_mesh_dir)
                elif style == "2ku":
                    player_2ku_dirs.append(player_mesh_dir)

    with open(os.path.join(output_dir, "data_rest_pose.txt"), "w") as f:
        f.writelines([line + '\n' for line in player_rest_dirs])
    with open(os.path.join(output_dir, "data_2ku.txt"), "w") as f:
        f.writelines([line + '\n' for line in player_2ku_dirs])

    # run prt_util
    try:
        cmds = []
        for data_dir in tqdm(player_rest_dirs+player_2ku_dirs):
            cmd = (
                "python -m apps.prt_util -i"
                f" {data_dir}"
                " -t nba"
            )
            cmds.append(cmd)
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        print(p.map(work, cmds))
    except Exception as e:
        print(f"{e}: {data_dir}")

    # run render_data
    try:
        dst_path = os.path.join(output_dir, 'rest_pose')
        cmds = []
        for data_dir in tqdm(player_rest_dirs):
            cmd = (
                "python -m apps.render_data -i"
                f" {data_dir}"
                f" -o {dst_path}"
                " -t nba -e"
            )
            cmds.append(cmd)
        dst_path = os.path.join(output_dir, '2ku')
        for data_dir in tqdm(player_2ku_dirs):
            cmd = (
                "python -m apps.render_data -i"
                f" {data_dir}"
                f" -o {dst_path}"
                " -t nba -e"
            )
            cmds.append(cmd)
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        print(p.map(work, cmds))
    except Exception as e:
        print(f"{e}: {data_dir}")

elif type == "rp":
    print("rp")
elif type == "twindom":
    print("twindom")
else:
    raise ValueError("Dataset type is currently not supported yet")
