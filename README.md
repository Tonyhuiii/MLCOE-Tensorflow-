# MLCOE-Tensorflow-
Tensorflow implementation of paper:  
Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (https://arxiv.org/abs/2208.09399).  
## Environment
● Ubuntu 20.04  
● NVIDIA RTX 3090
## Prerequisites
● Python 3.9.13  
● [Tensorflow](https://www.tensorflow.org/install) 2.10.0  
● cudatoolkit 11.2.2  
● cudnn 8.1.0 

## Fast experiment - Mujoco dataset 90% random missing
```
python train.py -c config/config_SSSDS4.json
python inference.py -c config/config_SSSDS4.json
```

## To do list
Part 1 (Dec. 15, 2022)  
● Tensorflow implementation of SSSD<sup>S4</sup>  (Finished on Nov.19)&#x2705;.    
Train and test on mujoco dataset with 90% RM in orginal config_SSSDSA.json.  
| Original paper | Pytorch code | Tensorflow code |
| :----:| :----: | :----: |
| 1.90(3)e-3 | [1.76e-3](figures/test_pytorch.png) | [1.67e-3](figures/test_tf.png) |     

● Tensorflow implementation of CSDI   

Part 2 Bonus question 1 (Jan. 7, 2023) (if possible)

Part 2 Bonus question 2 (Jan. 27, 2023) (if possible)




## Acknowledgments 
Code is based on Pytorch implementation of the original paper (https://github.com/AI4HealthUOL/SSSD).
