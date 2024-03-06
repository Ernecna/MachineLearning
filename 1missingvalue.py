import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#data upload
dataset = pd.read_csv('Dosyalar/eksikveriler.csv')

boy=dataset['boy']
boykilo=dataset[['boy','kilo']]

#print(dataset)

#missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

age=dataset.iloc[:,1:4].values
print(age)

## fit ile veriyi öğret tanıt
##transform ile değişikliği yap
imputer=imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])
print(age)

