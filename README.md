# DANN
A PyTorch implementation for Unsupervised Domain Adaptation by Backpropagation

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework (version 0.3.1)

Python 2.7/3.5

## Datasets
This responsity support Office-31 and Office-Home dataset 
```
cd src
python train_DANN.py --gpu_id 0 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_31_list.txt --t_dset_path ../data/office/amazon_31_list.txt --test_interval 500 --snapshot_interval 5000 --output_dir dann
```

```
You can set the command parameters to switch between different experiments. 
- "gpu_id" is the GPU ID to run experiments.
- "dset" parameter is the dataset selection. In our experiments, it can be "office" (for all the Office-31 tasks), "office-home" (for all the Office-Home tasks), "imagenet" (for task ImageNet->Caltech) and "caltech" (for Caltech->ImageNet).
- "s_dset_path" is the source dataset list.
- "t_dset_path" is the target dataset list.
- "test_interval" is the interval of iterations between two test phase.
- "snapshot_interval" is the interval of iterations between two snapshot models.
- "output_dir" is the output directory of the log and snapshot.
- "net" sets the base network. For details of setting, you can see network.py.
    - For AlexNet, "net" is AlexNet.
    - For VGG, "net" is like VGG16. Detail names are in network.py.
    - For ResNet, "net" is like ResNet50. Detail names are in network.py.
