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
 ```
python main.py  --data_name [DATASET] \
                --lr [LEARNING_RATE] \
                --scale_K [K] \ 
                --num_attention_heads [N_HEADS] \
                --train_name [LOG_NAME]
 ```
If you have downloaded the source codes, you can train the model just with data_name input.

