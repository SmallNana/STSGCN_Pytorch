# STSGCN_Pytorch
## This is a testing PyTorch version implementation of AAAI 2020. Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting
## ["AAAI 2020. Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting"](https://aaai.org/ojs/index.php/AAAI/article/view/5438/5294)
## Refer to the implementation of [Official](https://github.com/Davidham3/STSGCN)

## Requirements
* Python 3.8.10
* Pytorch 1.9.0+cu111
* Pandas 1.2.5
* Matplotlib 3.4.2
* Numpy 1.21.0
* seaborn 0.11.1
* 操作系统 CentOS Linux release 7.9.2009 (Core)

## issues
由于原作者并没有提供Pytorch版本的代码，出于学习的目的自己尝试实现了一下，可能存在问题，欢迎各位指正和讨论

## Usage
* 首先要用generate_datasets.py生成数据
```
python generate_datasets.py --output_dir ./data/processed/PEMS08/ --traffic_df_filename ./data/PEMS08/PEMS08.npz
``` 
``` 
python generate_datasets.py --output_dir ./data/processed/PEMS04/ --traffic_df_filename ./data/PEMS04/PEMS04.npz
``` 
* 训练用train.py，测试用test.py
``` 
python train.py
python test.py --checkpoint ./garage/PEMSD8/Val_MAE_17.96_best_model.pth
python test.py --checkpoint ./garage/PEMSD4/Val_MAE_20.91_best_model.pth
``` 
``` 
PEMSD4.conf 和 PEMSD8.conf 是训练与测试的参数配置文件，当使用这两个文件时，需要在train.py 和 test.py 文件中把 DATASET 改成要处理的数据集名
``` 
* 其他
``` 
data文件夹中的log是对训练和测试过程的记录
PEMSD4_history_vaild_MAE.npy 和 PEMSD8_history_vaild_MAE.npy 是训练过程中的指标数据的记录
如果没有GPU可能无法对 Val_MAE_17.96_best_model.pth 和 Val_MAE_20.91_best_model.pth 进行测试
``` 



