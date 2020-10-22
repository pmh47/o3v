
# Unsupervised object-centric video generation and decomposition in 3D


This repository contains the original implementation of the above [NeurIPS paper](https://arxiv.org/abs/2007.06705).
We include code for training the two variants O3V-voxel and O3V-mesh, as well as for generating our datasets.

[Project page](http://www.pmh47.net/o3v/)  -  [arXiv](https://arxiv.org/abs/2007.06705)

## Prerequisites

In a new conda env, run
```
conda install python=3.7 \
    tensorflow-gpu=1.15 tensorflow-probability=0.8 tensorflow-datasets=1.2 \
    matplotlib opencv scikit-image tqdm
pip install --no-deps git+https://github.com/pmh47/dirt
```


## Data Generation

### Rooms

1. In a new, separate conda env, change to the `gqn-dataset-generator` folder and run
   ```
   conda install python=3.6 tensorflow=1.15
   pip install -r requirements.txt
   ```
  
2. To generate data, change to `gqn-dataset-generator/opengl` folder and run
```
   python rooms_ring_camera.py --output-directory ../../data/gqn-rooms/1-4-obj_static_shadows_scale-1/train --image-size 80 --min-num-objects 1 --max-num-objects 4 --total-scenes 80000
   python rooms_ring_camera.py --output-directory ../../data/gqn-rooms/1-4-obj_static_shadows_scale-1/val --image-size 80 --min-num-objects 1 --max-num-objects 4 --total-scenes 10000
   python rooms_ring_camera.py --output-directory ../../data/gqn-rooms/1-4-obj_static_shadows_scale-1/test --image-size 80 --min-num-objects 1 --max-num-objects 4 --total-scenes 10000
```


### Traffic

1. Download and install carla 0.9.8
1. To generate raw data:
   1. create and activate a separate conda env with the packages in `requirements_carla.txt`
   1. run `pip install pygame==1.9.6`
   1. start the carla server in the background by running `/opt/carla/bin/CarlaUE4.sh -quality-level=Epic` (modify the path to carla if necessary)
    1. run 
    ```
    python carla_generation.py \
        --safe \
        --map Town02 \
        -e 5000 \
        --first-episode 0 \
        -o ./data/carla/raw/Town02-R19
    ```
1. To preprocess sub-sequences, run (in the original venv for O3V)
```
    python carla_preprocessing.py \
        --input-folder ./data/carla/raw/Town02-R19 \
        --output-folder ./data/carla/preprocessed/Town02-R19_fs-1_6-fr_1-4-obj \
        --num-observations-per-scene 6 \
        --frame-step 1 \
        --max-num-objects 4
```


## Training and Evaluation

To train O3V-voxel on **rooms**, run 
```
python train_rooms.py beta=1
```

To train O3V-voxel on **traffic**, run
```
python train_traffic.py \
    initial-beta=0.5 beta=2 beta-anneal-start=75000 beta-anneal-duration=25000
```

To train O3V-mesh on **rooms**, run
```
python train_rooms.py \
    obj-repr=mesh \
    pyr-levels=5 \
    obj-pres-bias=2 obj-min-pres=0.3 obj-min-pres-reg=1e2 \
    obj-l2-lapl=7.5 \ 
    initial-beta=0.5 beta=2 beta-anneal-start=15000 beta-anneal-duration=10000
```

To train O3V-mesh on **traffic**, run
```
python train_traffic.py \
    obj-repr=mesh \ 
    pyr-levels=5 \
    obj-min-pres=0.3 obj-min-pres-reg=10 \
    obj-min-pres-decay-start=100000 obj-min-pres-decay-duration=50000 \
    obj-l2-lapl=1e2
```

Evaluation (on the validation set) will occur automatically every 50K iterations during training (search the log for `subtotal statistics`), and images are regularly written to `./output`.
To run a standalone evaluation on the test set, use one of the above training commands, but append
```
restore=<run name>@<checkpoint iteration> \
final-eval=1
```
where `<run name>` is the name of the relevant subfolder of `./output`.

As noted in the paper, O3V-mesh is prone to local optima where object segmentation is very poor (particularly on on **traffic**); this is typically apparent early in training.
In this case, try running with a different random seed! 
