## MLCOE-Tensorflow-
Tensorflow implementation of paper:  
Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (https://arxiv.org/abs/2208.09399).  
### Environment
● Ubuntu 20.04  
● NVIDIA RTX 3090
### Prerequisites
● Python 3.9.13  
● [Tensorflow](https://www.tensorflow.org/install) 2.11.0  
● cudatoolkit 11.2.2  
● cudnn 8.1.0 

### To do list
#### Part 1 (Dec. 15, 2022)  
● Tensorflow implementation of SSSD<sup>S4</sup>  (Finished on Nov.19)&#x2705;.  
1）Train and test on mujoco dataset with 90% RM in config_SSSDS4.json (reproduce results in orginal paper).  
| Original paper | Pytorch code | Tensorflow code |
| :----:| :----: | :----: |
| 1.90(3)e-3 | [1.76e-3](figures/test_pytorch.png) | [1.67e-3](figures/test_tf.png) |    


Fast experiment - Mujoco dataset 90% random missing
```
python train.py -c config/config_SSSDS4.json
python inference.py -c config/config_SSSDS4.json
```

2） Train and test (MSE loss) on stock dataset with random missing in Length with all 6 features (finish experiment on Dec.1).  
| Hang Seng | Dow Jones | EuroStoxx |
| :----:| :----: | :----: |
| [1e-3](figures/Hang_Seng_test.png)| [4.9e-4](figures/Dow_Jones_29_test.png) | [8.9e-4](figures/EuroStoxx_47_test.png) |    

Fast experiment - Hang Seng dataset with missing_k=50
```
python train_stock.py -c config/config_SSSDS4_stock.json
python inference_stock.py -c config/config_SSSDS4_stock.json
```
Fast experiment - Dow Jones dataset with missing_k=30
```
python train_stock.py -c config/config_SSSDS4_dow.json
python inference_stock.py -c config/config_SSSDS4_dow.json
```
Fast experiment - EuroStoxx dataset with missing_k=20
```
python train_stock.py -c config/config_SSSDS4_euro.json
python inference_stock.py -c config/config_SSSDS4_euro.json
```

● Tensorflow implementation of CSDI   (Finished code on Nov.26, but have the bug keras optimizier didn't work, [the loss didn't decrease!!!](https://discuss.tensorflow.org/t/tensorflow-2-11-0-training-loss-doesnt-decrease/13281))


#### Part 2 Bonus question 1 (Jan. 7, 2023) (if possible)


#### Part 2 Bonus question 2 (Jan. 27, 2023) (if possible)   



### Acknowledgments 
Code is based on Pytorch implementation of the original paper (https://github.com/AI4HealthUOL/SSSD).
