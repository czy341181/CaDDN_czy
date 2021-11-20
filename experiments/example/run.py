import os


os.system("CUDA_VISIBLE_DEVICES=0,1 bash ./train.sh 2 kitti_example.yaml")

os.system("CUDA_VISIBLE_DEVICES=0 bash ./train.sh 1 kitti_example.yaml --e")