# Heart Disease prediction with kNN algorithms
### Introduction
This paper provides information on the appropriate application of the kNN algorithm to estimate patients' diagnoses of heart disease, taking into account the patient's 8 analyse results. All attributes (8) are numeric-valued. Two of these attributes are categorical. The data was collected from the Cleveland Clinic Foundation. "Num" is the target value of my model. Thus, if num = 1, the patient is more than 50% likely to have any heart disease, but if num = 0, the probability of having the disease is less than 50%. For the model, I used only 8 attributes of the dataset, which originally consisted of 14 attributes. I compared the accuracy between the two methods, using the kNN algorithm on the same number of (301) patients using the sklearn library (KNNeighboursClassifier) and writing entire algorithm from scratch. 

---

### Methodology

### 1) Importing Heart Disease Diagnose Dataset

In this procedure, I import the dataset collected and processed over Cleveland city through the pandas library. In the model I used these 8 attributes and deleted the other attributes:

1) age             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         5) chol

2) sex             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         6) thalach

3) cp  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                     7) thal

4) trestbps           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      8) oldpeak


```python
import pandas as pd
df_cleveland = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
df_cleveland.drop(df_cleveland.columns[[5,6,8,10,11]], axis = 1, inplace = True)
```

Although there is no need for calculation and subsequent steps, for manual debugging purposes, I entered the column names manually using the *pandas.rename()* method, as there is no section in the dataset file that contains the attribute names.

```python
df_cleveland.rename(columns = {'63.0':'age', '1.0':'sex', '1.0.1':'cp', '145.0':'trestbps', '233.0':'chol', '150.0':'thalach', '6.0':'thal', '2.3':'oldpeak','0':'num' }, inplace = True)
```

Although the KNN algorithm works seamlessly on multi-classes datasets, I made the target attribute a binary class due to a sharp drop in accuracy when applied the algorithm to this dataset. I also deleted 3 records because there are missing values in the “thal” attribute.

```python
df_cleveland['num'] = df_cleveland['num'].map({0:'0', 1:'0',2:'1',3:'1'})
df_cleveland = df_cleveland.dropna(subset=['num'])
df_cleveland['num'] = df_cleveland['num'].astype(int)

df_cleveland.drop(df_cleveland[df_cleveland['thal'] == '?'].index, inplace = True)
df_cleveland['thal'] = df_cleveland['thal'].astype(float)

```

Finally, I separated the target and input attributes for the calculations in the next steps

```python
x = df_cleveland.iloc[:, 0:-1]
y = df_cleveland.iloc[:,-1]
```

### 2)Writing KNN algorithm from scratch

In my algorithm, I used pandas for dataframe manipulation, sklearn.model_selection to split the dataset, sklearn.metrics for evaluation, and numpy libraries for array calculations.

First, the dataset will be split to perform calculations on it. Although the entire dataset for the KNN algorithm can be used as both a test and a train at the same time, I split the dataset to make it easier to calculate accuracy and fit for the application. So I set aside 70% of the dataset as a train and 30% as a test dataset.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)
```

I determined the distance between the data points using euclidian distance. However, as you know, due to the fact that two attributes in our dataset are categorical and the euclidian distance is not optimal method for calculating distance, it will affect the accuracy of our model.

```python
import numpy as np
def euc_dist(p1, p2):
    return np.linalg.norm(p1-p2)
```

 

The part of my algorithm that calculates the results is the following knn_scratch function:

```python

from scipy.stats import mode
def knn_scratch(X_train, X_test, y_train,  k):
    y_hat = []
    for test_p in X_test.to_numpy():
        distances = []
        for i in range(len(X_train)):
            distances.append(euc_dist(X_train.to_numpy()[i], test_p))

        distance_df = pd.DataFrame(data = distances, columns = ['distance'],index = X_train.index) 
        kNN_distances = distance_df.sort_values(by = ['distance'], axis = 0)[:k]
        targets = y_train.loc[kNN_distances.index]
        labeling = mode(targets).mode[0]           
        y_hat.append(labeling) 
    return y_hat

y_hat_pred = knn_scratch(X_train, X_test, y_train, k)
```

**Explanation of Algorithm**

1)  First a point is selected in the X_test dataset (*test_p*). 

![image](https://user-images.githubusercontent.com/58222828/173659514-8c6e4f34-57ce-4774-ae61-b874c2518467.png)

2) The distance between all data points in the X_train dataset is calculated from the selected test_p and each result is thrown into the distances array

3) To import the y_train labels in the corresponding index from the elements in the Distances array, we convert them to dataframe using pd.Dataframe () and implement the indexes from X_train.

![image](https://user-images.githubusercontent.com/58222828/173659666-44963f43-e77c-4446-8737-ba9f03ad0b86.png)

4)To select the nearest K-number points, we sort the dataframe we prepared above and throw the first K-number element into a new variable - the "kNN_distances" .

![image](https://user-images.githubusercontent.com/58222828/173659830-e28fae56-5376-451e-b34e-3597e5e7af3e.png)

5) As a final step, we extract the labels from the corresponding indices in the *y_train* dataset to the indices of the distances of the number K, find the most frequent ones using *scipy.stats.mode()* and throw them into the *y_hat* array.

![image](https://user-images.githubusercontent.com/58222828/173659900-3b357db0-8d50-40e1-8589-e48a6deaf3e8.png)

### 4) İmporting KNeighborClassifier and evaluating the models

In this procedure, I tried to find the optimal k for both models. To compare the results of my model, I import the KNeighborsClassifier module and fit  model by using X_train and y_train datasets for a given k and splitting  percentage. I apply the same procedure to the *knn_scratch()* function and get the accuracy for 30 number of K's. I will discuss the results in the next section.

```python
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
external_k_results = []
my_k_results = []
max_k = 30

for i in range(1,max_k):
    y_hat_pred = knn_scratch(X_train, X_test, y_train, k = i)
    
    external_model = KNeighborsClassifier(n_neighbors=i)
    external_model.fit(X_train,y_train)
    external_model_pred = external_model.predict(X_test)
    
    my_k_results.append(accuracy_score(y_hat_pred, y_test))
    external_k_results.append(accuracy_score(external_model_pred, y_test))

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]    
plt.plot(range(1,max_k+1), my_k_results, color = 'orange', marker = 'o')
plt.plot(range(1,max_k+1), external_k_results, color = 'blue', marker = '*', linestyle='dashed')

print('\nMY MODEL: ')
max_accuracy = np.amax(my_k_results)
print('best accuracy: ' + str(max_accuracy))
print('optimal k=' + str(my_k_results.index(max_accuracy)+1))

print('\nKNeighborsClassifier ')
max_accuracy = np.amax(external_k_results)
print('best accuracy: ' + str(max_accuracy))
print('optimal k=' + str(external_k_results.index(max_accuracy)+1))
```

---

## Results & Discussion

### Results of KNeighborsClassifier

```python
#best accuracy: 0.8160919540229885
#optimal k=15
```

![image](https://user-images.githubusercontent.com/58222828/173659986-d509cbf4-f5d7-42d9-a231-402717b35693.png)

### Results of my model

```python
#best accuracy: 0.8160919540229885
#optimal k=15
```

![image](https://user-images.githubusercontent.com/58222828/173660106-cef7ed74-3464-48e7-93f0-7bef35568014.png)

As you can see, when calculating using 8 attributes, we get the same accuracy in each k for both models. In practice, if the kNN algorithm is used for this dataset, the scratch model can be used instead of the sklearn library to save computational power. It will also be time consuming to simply iterate 20 numbers of K during the prediction so that we get stable accuracy after 20 neighbors. However, as I stated in Section 2, the accuracy of my model may differ from KNeighborsClassifier because of the categorical attributes. Thus, my model only calculates the euclidian distance between points. So when we reduce the number of attributes and keep the "sex" and "cp" attributes, result will be:

![image](https://user-images.githubusercontent.com/58222828/173660955-4531324c-0667-4758-9786-9a3bfc1e46e3.png)

![image](https://user-images.githubusercontent.com/58222828/173660986-5b7be481-4cc9-4c33-a33f-88fed920b503.png)
                            blue dashed lines - KNeighborClassifier &nbsp;&nbsp;&nbsp;&nbsp;
                            yellow line - My Model

Thus, although the optimal k and accuracy for this split are the same for both models, there is a difference of 2% -5% on the best accuracy between the different splits. In most splits, the average accuracy for each k is 0.5% to 2% greater in the KNeighborsClassifier. Due to the context of the dataset, no significant change in accuracy is observed when trying to differentiate the models from the number of false-negatives in the confusion matrix. Also, when we delete two categorical attributes from the calculation, we encounter an increase in accuracy for both models.
