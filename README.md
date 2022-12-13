# MLCOE-Tensorflow-implementation
Tensorflow implementation of paper: Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models (https://arxiv.org/abs/2208.09399).  
## Environment
● Ubuntu 20.04  
● NVIDIA RTX 3090
## Prerequisites
● Python 3.9.13  
● [Tensorflow](https://www.tensorflow.org/install) 2.11.0  
● cudatoolkit 11.2.2  
● cudnn 8.1.0 

## To do list
### Part 1 (Dec. 15, 2022)  
#### ●  Tensorflow implementation of SSSD<sup>S4</sup> (finished on Nov.19)&#x2705;.  
***1） Train and test on MuJoCO dataset with 90% RM in config_SSSDS4.json (reproduce results in orginal paper).***    
*Note: some limitations in the original PyTorch code*            
1.the training batch is fixed during the iteration (they didn't use PyTorch Dataset and Dataloader for random shuffle);     
2.the random misssing mask for different batchs is duplicated in the same iteration (not random enough);

| Original paper | PyTorch code | Tensorflow code |
| :----:| :----: | :----: |
| 1.90(3)e-3 | [1.76e-3](figures/test_pytorch.png) | [1.67e-3](figures/test_tf.png) |    


Fast experiment - MuJoCO dataset 90% random missing
```
python train.py -c config/config_SSSDS4.json
python inference.py -c config/config_SSSDS4.json
```

***2) Train and test on stock dataset with blackout missing (BM) with all 6 features (finished on Dec.1).***     
*Note: some improvements in train_stock.py*         
1.using different masks for each batch in the same iteration;     
2.add my_loss function, which counts nonzero numbers in the conditional mask (imputation noise), same as the orginal PyTorch version using index for valid imputation noise (z[loss_mask]). Original mse loss (tf verison train.py) directly count all the mask numbers, although the value is zero for conditional noise (z*loss_mask).     

*Stock data download and preprocess*               
1.Take Hang_Seng for example (stock_data/data.py), download data with tickers, check valid trading days (over 10 years), save 10year stock.txt;     
2.Iterate the weekdays from start to end, mask the trading days with holiday:-1; nan:0; valid:1;     
3.Normalize the raw data with min-max, scale to the [0,1] for each feature, and drop most of the nan data for stocks not on the market in early days;    
4.Split into the training dataset (0.8) and testing dataset (0.2).    

| Dataset (iteration, batch, length, feature)| Hang Seng | Dow Jones |  EuroStoxx |
| :----:| :----: | :----: |  :----: |
| training size | (78, 13, 239, 6) | (49, 42, 137, 6) | (61, 41, 94, 6) |    
| testing size| (11, 23, 239, 6) | (6, 87, 137, 6) | (5, 125, 94, 6) |        

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
*Tesing results (MSE loss)*    
| Hang Seng | Dow Jones | EuroStoxx |
| :----:| :----: | :----: |
| [1e-3](figures/Hang_Seng_test.png)| [4.9e-4](figures/Dow_Jones_29_test.png) | [8.9e-4](figures/EuroStoxx_47_test.png) |    


#### ● Tensorflow implementation of CSDI   (finished code on Nov.26)
*Bug: keras optimizier didn't work, the loss didn't decrease. (struggling!!)*              

***1) 20% RM on PTB-XL (CSDI) (updated on Dec.13)***     
*Note: cannot reproduce the results using orginal PyTorch code with same training config.*         
1.confusing masking config: In CSDI PyTorch code modified by SSSD authors, the code for **RM, MNR, BM** initlization in dataset are added, but the masks will not change in training. However, the original code for **random strategy** (missing ratios [0%, 100%]) or **historical strategy** in CSDI paper is still maintained in the CSDI model, which will change the mask during the training.        
2.data length: should be 1000 or 250? For PTB-XL 1000 dataset , **considered L = 250 time steps** is mentioned in the paper. However, the table in the original paper shows training batch 4 with sample length 1000. Using this dataset config, **NVIDIA RTX3090 24GB out of memory** , which shoulde be same for NVIDIA A30 cards with 24GB that author used. Therefore, data length is set to 250, batch size is set to 16 for model training.

| Model | MAE | RMSE |  CRPS |
| :----:| :----: | :----: |  :----: |
| Paper| 0.0038±2e-6 | 0.0189±5e-5 | 0.0265±6e-6 |    
| PyTorch (20% RM + Random strategy)| [0.0102](figures/rm_0.2.png) | 0.0514 | 0.0698| 
| PyTorch (0% RM + fixed 20% Random strategy)|  | | | 
| Tensorflow| still |debug | ing | 

### Part 2 Bonus question  (if have time after finishing part 1)
● Bonus question 1 (Jan. 7, 2023)       
● Bonus question 2 (Jan. 27, 2023)



## Acknowledgments 
Code is based on Pytorch implementation of the original paper (https://github.com/AI4HealthUOL/SSSD).
