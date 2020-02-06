# Histoplathologic Cancer Detection

This repo is my solution for the Kaggle Challenge.

## How to Run the code

Download the dependencies first:

- Albumentations
- cv2
- Pillow
- Pytorch (supports v1.1x)
- Numpy 
- Pandas
- Scikit-Learn
- Scipy

### For training

Specify the `path/to/dataset/` in `--path`

#### For using pretrained model on ImageNet
```bash
python train.py --path --seed --batch_size --num_workers --epochs --lr --eval_every
```

#### For training from scratch
```bash
python train.py --path --seed --batch_size --pretrained --num_workers --epochs --lr --eval_every
```

### For Inference

The trained models will be saved in `saved_models` directory.

#### Infering without TTA
```bash
python infer.py --path --model_name --batch_size --seed --num_workers --model_path --ckpt_name

```
#### Infering with TTA
```bash
python infer.py --path --model_name --batch_size --seed --num_workers --model_path --ckpt_name --use_tta --num_tta
```


## Results

The leader board scores are tabulated as below:

Model | Public LB | Private LB
:---: | :---: | :---: 
Densenet169v1 | 0.4127 | 0.5124
Densenet169v2 | 0.4155 | 0.5120
Densenet169v3 | 0.4145 | 0.5137
Densenet169v4 | 0.4383 | 0.5078
Resnet50 | 0.9625 | 0.9609
Resnet50 ( TTA) | 0.9604 | 0.953
CBAM_Resnet50 | 0.9711 | 0.9742
CBAM_Resnet50 (TTA) | 0.9738 | 0.9764

- The various Densenet169 versions refer to various TTA strategies and ensembles used.
- Regarding CBAM Resnet50, TTA is used, but no ensemble.

Kaggle's GPU quota restricted extensive experimentation.


Note : For CBAM only ResNet50 pretrained model is available. For rest, you need to train from scratch.
