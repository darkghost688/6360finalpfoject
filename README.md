# Final Project


## Final Project by zhiwen zhong
*  FINAL Project: "Organize Biological Data through the model built project"
  
git clone 
open nifH.ipynb

run the code:

```python
pip install pandas matplotlib seaborn numpy scikit-learn biopython
```

### what we have learned from the class:
* the philosophy of data science and linux
* how to control a computer from the command line
* the basics of 3 computer languages that are critical in processing, analyzing, and visualizing biological data
  * `bash`
  * `R`
  * `python`
* version controling our work using git
* distributing our work using GitHub
* the final project that professor want is we can organize Biological Data by this class
* so just build a model try to use the sequenced genes

# * Main purpose
### Organize Biological Data by models created based on the 20 nifH genes data to predict which species of the sample belong to.


* Download the data from :
### NCBI

![alt text](https://p.ipic.vip/k0o1dy.png)

and through 5 different models to predict the accuracy.

# this project shows：
### 1 Batch processing of data
If the data sources are scattered, how to unify them in batches
![alt text](https://p.ipic.vip/go8ge3.png)
### 2 How to convert gene format into usable data format

### 3 How to aggregate data into the same table
![alt text](https://p.ipic.vip/96je1o.png)
### 4 How to combine valid data into one table
![alt text](https://p.ipic.vip/9xcmnf.png)
### 5 How to process and draw the data in the table
1 hot map
![alt text](https://p.ipic.vip/2a6wf0.png)
2 Scatter plot
![alt text](https://p.ipic.vip/fdfv7z.png)
3 Line chart
![alt text](https://p.ipic.vip/mi4mjx.png)




# Highlights

## use the 5 models to show the way to handle data

### 1 Lasso Regression model

### 2 Ridge Regression

### 3 Randon Forest

### 4 Gradient Boosting Model (GBM)

### 5 SVM Model

### 6 KNN Model

### 7 Neural Network


details:

Do you know that there is a very interesting function in ncbi?
It is called Blast

We can compare the genes we get after sequencing with the database, so how does this function work?
I have a very interesting idea

If you can write an algorithm yourself, you can compare it with the nifH gene in the database, and then determine whether it belongs to this species

Because everyone knows that the core of sending the samples for sequencing is to know how many probabilities our samples have for this, how many probabilities they have for that, and how many probabilities they are not what we want

Although we can import packages in R to achieve this, how is this done?

So can we use this project to replicate a project

So I want to reproduce it with the technology I have learned

So, let's think about how to do it with the technology we have learned now

# characteristic：

## ID: Unique identifier of the gene sequence.

## Sequence: The base sequence of the gene (A, T, C, G).

## Length: The length of the gene in base pairs.

## GC_Content: GC content, which is (G + C) / total length.

## Sequence entropy (Shannon entropy)

## GC content

## ATCG share

## Maximum base repeat length


# FINAL RESULT
Model Performance Comparison:
           Model     RMSE        R²
Lasso Regression 6.981998 -0.163753
   Random Forest 6.116613  0.106853
             SVM 6.970041 -0.159770
             KNN 6.873742 -0.127944
  Neural Network 6.344727  0.038992
