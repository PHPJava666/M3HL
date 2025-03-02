

## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.3 and Python 3.8.10. All experiments in our paper were conducted on NVIDIA Quadro RTX 6000 24GB GPU with an identical experimental setting.

## Usage

To train a model,
```
python ./code/ACDC_M3HL_train.py  #for ACDC training
```
```
python ./code/LA_M3HL_train.py  #for LA training
```

To test a model,
```
python ./code/test_ACDC.py  #for ACDC testing
```
```
python ./code/test_LA.py  #for LA testing
```

