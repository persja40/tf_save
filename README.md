# Requirements
python 3.5+ <br>
tensorflow(-gpu CUDA 9.0 & cuDNN 7.0 required) <br>
numpy <br>
matplotlib

# How to run
Training and saving NN
----
```bash
python .\train_save.py <function> <limit_min> <limit_max> <step> <learning_data_ratio>
```
### Example
```bash
python .\train_save.py '2*sin(1.5*x-5)*cos(-3.2*x+1.7)' 0 10 0.1 0.7
```
Loading & using NN from file
---
```bash
python .\load.py <function> <limit_min> <limit_max> <step> <learning_data_ratio> <path_to_model>
```
### Example
```bash
python .\train_save.py '2*sin(1.5*x-5)*cos(-3.2*x+1.7)' 0 10 0.1 0.7 './saved.model/model_save'
```
