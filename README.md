## Distributed Arcface Training in Pytorch

The "arcface_torch" repository is the official implementation of the ArcFace algorithm. It supports distributed and sparse training with multiple distributed training examples, including several memory-saving techniques such as mixed precision training and gradient checkpointing. It also supports training for ViT models and datasets including WebFace42M and Glint360K, two of the largest open-source datasets. Additionally, the repository comes with a built-in tool for converting to ONNX format, making it easy to submit to MFR evaluation systems.


## Requirements

To avail the latest features of PyTorch, we have upgraded to version 1.12.0.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) (torch>=1.12.0).
- `pip install -r requirement.txt`.
  
## How to Training

To train a model, execute the `train_v2.py` script with the path to the configuration files. The sample commands provided below demonstrate the process of conducting distributed training.

### 1. To run on one GPU:

```shell
python train_v2.py configs/ms1mv3_r50_onegpu
```

Note:   
It is not recommended to use a single GPU for training, as this may result in longer training times and suboptimal performance. For best results, we suggest using multiple GPUs or a GPU cluster.  


### 2. To run on a machine with 2 GPUs:

```shell
torchrun --nproc_per_node=2 train_v2.py configs/ms1mv3_r50
```


## Download Datasets or Prepare Datasets  
- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)
- [Your Dataset, Click Here!](docs/prepare_custom_dataset.md)

Note: 
If you want to use DALI for data reading, please use the script 'scripts/shuffle_rec.py' to shuffle the InsightFace style rec before using it.  
Example:

`python scripts/shuffle_rec.py ms1m-retinaface-t1`

You will get the "shuffled_ms1m-retinaface-t1" folder, where the samples in the "train.rec" file are shuffled.

















