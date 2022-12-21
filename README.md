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
***1） Validate the tensorflow code on MuJoCo dataset with config_SSSDS4.json (90%RM), config_SSSDS4_mujoco_70.json(70%RM).***    
*Note: some limitations in the original PyTorch code*            
1.the training batch is fixed during the iteration (they didn't use PyTorch Dataset and Dataloader for random shuffle);     
2.the random misssing mask for different batchs is duplicated in the same iteration (not random enough);

Fast experiment - MuJoCo dataset 90% random missing
```
python train.py -c config/config_SSSDS4.json
python inference.py -c config/config_SSSDS4.json
```

*IMPUTATION MSE RESULTS (RM 90% + Iterations 150,000)* (reproduce MSE results in orginal paper)&#x2705;    
| Original paper | PyTorch code | Tensorflow code |
| :----:| :----: | :----: |
| 1.90(3)e-3 | [1.76e-3](figures/test_pytorch.png) | [1.67e-3](figures/test_tf.png) |      

*IMPUTATION MSE RESULTS (RM 70%)* (reproduce MSE results in orginal paper)&#x2705; 
| Original paper (Iterations 232,000) | Tensorflow code (Iterations 240,000)|
| :----:| :----: |
| 0.59(8)e-3 | [0.53e-3](figures/test_tf_24000_rm.png) |

***2) Train and test on stock dataset with random missing in length with all 6 features (finished on Dec.1).***     
*Note: some improvements in train_stock.py*         
1.using different masks for each batch in the same iteration;     
2.add my_loss function, which counts nonzero numbers in the conditional mask (imputation noise), same as the original PyTorch version using index for valid imputation noise (z[loss_mask]). In the tensorflow verison of train.py, original mse loss directly counts all the mask numbers, although the value is zero for conditional noise (z*loss_mask).     

*Stock data download and preprocess*               
1.take Hang_Seng for example (stock_data/data.py), download data with tickers, check valid trading days (over 10 years), save 10year stock.txt;     
2.iterate the weekdays from start to end, mask the trading days with holiday:-1; nan:0; valid:1;     
3.normalize the raw data with min-max, scale to the [0,1] for each feature, and drop most of the nan data for stocks not on the market in early days;    
4.split into the training dataset (0.8) and testing dataset (0.2).    

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
*IMPUTATION MSE RESULTS*    
| Hang Seng | Dow Jones | EuroStoxx |
| :----:| :----: | :----: |
| [1e-3](figures/Hang_Seng_test.png)| [4.9e-4](figures/Dow_Jones_29_test.png) | [8.9e-4](figures/EuroStoxx_47_test.png) |    


#### ● Tensorflow implementation of CSDI   (finished code on Nov.26)
*Bug: the training loss didn't decrease. (solved on Dec.21, correct the mistake for tensor shape not changing after the transformer encoder layer)*              

***1) 20% RM on PTB-XL (CSDI) (updated on Dec.13)***     
*Note: cannot reproduce the results using original PyTorch code with same training config.*         
1.data length: should be 1000 or 250? For PTB-XL 1000 dataset, **considered L = 250 time steps** is mentioned in the paper. However, the table in the original paper shows training batch 4 with sample length 1000. Using this dataset config, **NVIDIA RTX3090 24GB out of memory** , which should be same for NVIDIA A30 cards with 24GB memory that author used. For training config, sample length is set to 250, batch size is set to 32.       
2.confusing masking config: In CSDI PyTorch code modified by SSSD authors, the code for **RM, MNR, BM** initialization in dataset are added, but the masks will not change in training. Also I found the RM code is not same as the RM definition in SSSD paper and SSSD code. However, the original code for **random strategy** (missing ratios [0%, 100%]) or **historical strategy** in CSDI paper is still maintained in the CSDI model, which will change the mask during the training. I have tried the two combination of the training settings in original code.    
| Results | Config | MAE | RMSE|  CRPS |
| :----: | :----:| :----: | :----: |  :----: |
| Paper | 20% RM| 0.0038±2e-6 | 0.0189±5e-5 | 0.0265±6e-6 |    
| PyTorch |  [20% RM + Random strategy](figures/rm_0.2.png)| 0.0102 | 0.0514 | 0.0698| 
| PyTorch |  [0% RM + fixed 20% Random strategy](figures/fixed_0.2_RS.png)| 0.0114 | 0.0351| 0.0783 |      
                          
3.I reimplemented the **RM, MNR, BM** in the CSDI training module referenced from the masking code in the SSSD, the results are shown in [1],[2],[3]. For CSDI evaluate code, I noticed that it used the median value of 10 samples generated for each test sample to calculate mae and rmse instead of averaged value as in the SSSD paper. I recalculate the averaged value based on the saved generated samples, the MAE and RMSE results are represented by average(median) in Table.
| Results | Config | MAE | RMSE|  CRPS |
| :----: | :----:| :----: | :----: |  :----: |
| Paper | 20% RM| 0.0038±2e-6 | 0.0189±5e-5 | 0.0265±6e-6 |    
| PyTorch [1]|  [20% RM implemented](figures/rm_newversion_0.2.png)  | 0.0168(0.0139) | 0.0470(0.0390)| 0.0928 | 
| Tensorflow| | still |debug | ging | 
| Paper  | 20% MNR| 0.0186±1e-5 | 0.0435±2e-4 | 0.1306±5e-5 | 
| PyTorch [2] | [20% MNR implemented](figures/csdi_nrm.out)| 0.0158(0.0136) | 0.0553(0.0397)| 0.0918 | 
| Paper  | 20% BM| 0.1054±4e-5 | 0.2254±7e-5 | 0.7468±2e-4 |   
| PyTorch [3]| [20% BM implemented](figures/csdi_bm.out)| 0.0590(0.0486) | 0.1257(0.1084)| 0.3278 | 


### Part 2 Bonus question  (if have time after finishing part 1)
● Bonus question 1 (Jan. 7, 2023)       
● Bonus question 2 (Jan. 27, 2023)



## Acknowledgments 
Code is based on Pytorch implementation of the original paper (https://github.com/AI4HealthUOL/SSSD).
