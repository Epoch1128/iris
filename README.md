Perceptron and Naive Bayes for iris classification
===========================

****
	
|Author|Gao Chaofei|
|---|---
|My mailbox|18376047@buaa.edu.cn


****
## Contents
* [Dataset](#Dataset)
* [Preprocess](#Preprocess)
* [Usage](#Usage)
    * Perceptron
    * Naive Bayes
* [Result](#Result)

## Dataset
------
Our dataset is from iris. You can find it in ../data/iris.txt.


## Preprocess
------
### use data_pre.py to do preprocessing
```
python data_pre.py
```
After that, you will get train_data.npy and val_data.npy


## Usage
-----
### For perceptron
```
python perceptron.py
```
### For Bayes
```
python bayes.py
```

## Result
-----
|Method|Result|
|----|-----|
|Perceptron|1.0|
|Naive Bayes|0.98|



