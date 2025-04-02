# MSTRec


This is the source code for our Paper 'Contrastive Enhanced Multi-Scale Transformer for Sequential
Recommendation'



# Overview
MSTRec is a Transformer architecture, wherein each block starts with a multi-scale attention encoder and ends with a feed-forward layer.  The core module of our MSTRec is the multi-scale attention encoder, which enables the model to present multi-scale periodic patterns adaptively and capture multi-scale periodic patterns explicitly. Essentially, we enhance the original self attention with multi-scale attention projection, thereby extending the Transformer's capability to capture key dynamics with different periodic patterns.  

![Framework](images/model.jpg)

# Datasets
We utilize four benchmark datasets to evaluate our MSTRec, all of which can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1Ir0nVoC_1flw3zW9N_ANck_XaGTvCNTa): 
* Amazon Beauty, Sports are two representative sub-datasets gathered from Amazon dataset, which contains a series of product reviews crawled from Amazon.com. They are split by the top-level product categories on Amazon, we adopt the “Beauty”, “Sports and Outdoors (Sports)” categories.
* ML-1M is a large and dense dataset with long item sequences, which collected from the movie recommendation site MovieLens. 
* Yelp is a large business recommendation dataset. We treat the transaction records after $\text{January}$ $1 ^ {st}$,  $2019$.
* Note that the `*_same_target.npy` files in [Google Drive link](https://drive.google.com/drive/folders/1Ir0nVoC_1flw3zW9N_ANck_XaGTvCNTa) for the four datasets are utilized for training DuoRec, FEARec and our MSTRec, both of which incorporate contrastive learning.


# Environment Setting
 ```
conda create -n MSTRec python=3.8
conda activate MSTRec
 ```
The required environment settings are detailed in the `requirements.txt` file.



# Quick-Start

## How to train MSTRec
If you have downloaded the source codes, you can just `run main.py` train the model：
 ```
python main.py  --data_name [DATASET] \
                 --lr [LEARNING_RATE] \
                 --scale_K [K] \ 
                 --num_attention_heads [N_HEADS] \
                 --hidden_dropout_prob [DROPOUT] \
                 --train_name [LOG_NAME]
 ```
 `train_name`： name for log file and checkpoint file.


Example for Beauty
 ```
python main.py  --data_name Beauty \
                 --lr  0.001\
                 --scale_K 3 \ 
                 --num_attention_heads 2 \
                 --hidden_dropout_prob 0.5 \
                 --train_name MSTRec_Beauty
 ```


* Note that trained model (.pt) and train log file (.log) will saved in  `src/output` folder.
* The parameters of our model with contrastive learning remain the same as those of the baselines DuoRec and FEARec.


## How to test MSTRec
If you have trained MSTRec on a certain dataset, you can test the trained model：
 ```
python main.py  --data_name [DATASET] \
                 --scale_K [K] \ 
                 --num_attention_heads [N_HEADS] \
                 --load_model [TRAINED_MODEL_NAME] \
                 --do_eval
 ```

* Note that trained model (.pt) must be in `src/output`： name for log file and checkpoint file
* `load_model`:  trained model name without .pt

Example for Beauty
 ```
python main.py  --data_name Beauty \
                 --scale_K 3 \ 
                 --num_attention_heads 2 \
                 --load_model MSTRec_Beauty \
                 --do_eval
 ```
