# damNN-vslam
Dense Accurate Map Building using Neural Networks

## Setup environment
```
git clone https://github.com/mech0ctopus/damNN-vslam.git
cd damNN-vslam
conda create --name damnn --file requirements.txt
conda activate damnn
```
## Download Training Data
[Download Pre-processed RGB and Depth Images (Re-sized and colorized) Training Images (5.5GB)](https://mega.nz/file/O1sn3TQQ#fbXlhG5T8Ad30CTtfwvKyKfgDyH3Aa2tq_fSoYhTA0U)

[Download KITTI Odometry Ground Truth (4MB)](http://www.cvlibs.net/download.php?file=data_odometry_poses.zip)

Note: Raw image data is from the [KITTI Raw Dataset (synced and rectified)](http://www.cvlibs.net/datasets/kitti/raw_data.php) and the [KITTI Depth Prediction Dataset (annotated depth maps)](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

## Train Models
```
python damnn-vslam.py
```

## View Results in Tensorboard
```
cd damNN-vslam
tensorboard --logdir logs
```
