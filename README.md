# M3HL


## Introduction
Official code for "M3HL".


## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.3 and Python 3.8.10. All experiments in our paper were conducted on NVIDIA Quadro RTX 6000 24GB GPU with an identical experimental setting.
## Datasets
**Preprocess**: refer to the image pre-processing method in [CoraNet](https://github.com/koncle/CoraNet) and [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) for the Pancreas dataset, Left atrium and ACDC dataset. 
The `dataloaders` folder contains the necessary code to preprocess the Left atrium and ACDC dataset. 
Pancreas pre-processing code can be got at [CoraNet](https://github.com/koncle/CoraNet).

**Dataset split**: The `./Datasets` folder contains the information about the train-test split for all three datasets.
## Usage
We provide `code`, `data_split` and `models` (Include pre-trained models and fully trained models) for Pancreas, LA and ACDC dataset.

Data could be got at [Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT), [LA](https://github.com/yulequan/UA-MT/tree/master/data) and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

To train a model,
```
python ./code/LA_train.py  #for LA training
``` 

To test a model,
```
python ./code/test_LA.py  #for LA testing
```

## Acknowledgements
Our code is largely based on [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) and [SS-Net](https://github.com/ycwu1997/SS-Net). Thanks for these authors for their valuable work, hope our work can also contribute to related research.
