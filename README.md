# M3HL


## Introduction
Official code for "M3HL".


## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.3 and Python 3.8.10. All experiments in our paper were conducted on NVIDIA Quadro RTX 6000 24GB GPU with an identical experimental setting.
## Datasets
**Preprocess**: refer to the image pre-processing method in [CoraNet]and [BCP] for the Left atrium dataset and ACDC dataset. 
The `dataloaders` folder contains the necessary code to preprocess the Left atrium and ACDC dataset. 
                                                                                 
**Dataset split**: The `./Datasets` folder contains the information about the train-test split for all datasets.
## Usage
We provide `code`, `data_split` and `models` (Include pre-trained models and fully trained models) for the Left atrium dataset and ACDC dataset.


To train a model,
```
python ./code/ACDC_train.py  #for ACDC training
``` 

To test a model,
```
python ./code/test_ACDC.py  #for ACDC testing
```

## Acknowledgements
Our code is largely based on [BCP] and [SS-Net]. Thanks for these authors for their valuable work, hope our work can also contribute to related research.
