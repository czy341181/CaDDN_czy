# CaDDN

This is the unofficial implementation of the paper CaDDN for personal study.
This codebase is based on https://github.com/xinzhuma/monodle.

## Usage

### Installation
This repo is tested on our local environment (python=3.7, cuda=10.1, pytorch=1.4)

```bash
conda create -n CaDDN python=3.7
```
Then, activate the environment:
```bash
conda activate CaDDN
```

Install PyTorch:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt

python setup.py develop
```



### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/
      |object/			
        |training/
          |calib/
          |image_2/
          |velodyne/
          |label/
        |testing/
          |calib/
          |image_2/
          |image_3/
```

### Training

Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 CUDA_VISIBLE_DEVICES=0 python ../../tools/train_val.py --config kitti_example.yaml
```

[comment]: <> (### Plan)

[comment]: <> (We want to build a codebase based on the image voxel method &#40;DSGN&#40;stereo&#41;, CaDDN&#40;monocular&#41;, SECOND&#40;LiDAR&#41;&#41;.)

[comment]: <> (The implementation of CaDDN is coming soon.)

[comment]: <> (The implementation of Voxel-based code-based is coming soom.)

