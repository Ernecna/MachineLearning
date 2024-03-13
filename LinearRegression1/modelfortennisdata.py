"" #1.kutuphaneler
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    #2.veri onisleme
    #2.1.veri yukleme
    veriler = pd.read_csv('Dosyalar/odev_tenis.csv')
    #pd.read_csv("veriler.csv")
    #test
    print(veriler)
        
    
    #encoder: Kategorik -> Numeric
    play = veriler.iloc[:,-1:].values
    print(play)
    
    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    
    play[:,0] = le.fit_transform(play[:,0])
    
    print(play)
    
    #encoder: Kategorik -> Numeric
    windy = veriler.iloc[:,-2:-1].values
    print(windy)
    
    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    
    windy[:,0] = le.fit_transform(windy[:,0])
    
    print(windy)
    ohe = preprocessing.OneHotEncoder()
    windy= ohe.fit_transform(windy).toarray()
    print(windy)
    
    
    
    #encoder: Kategorik -> Numeric
    outlook = veriler.iloc[:,0:1].values
    print(outlook)
    
    from sklearn import preprocessing
    
    le = preprocessing.LabelEncoder()
    
    outlook[:,0] = le.fit_transform(outlook[:,0])
    
    print(outlook)
    
    
    ohe = preprocessing.OneHotEncoder()
    outlook = ohe.fit_transform(outlook).toarray()
    print(outlook)
    
    
    son=pd.DataFrame(data=outlook,index=range(len(outlook)),columns=['o','r','s'])
    enson=pd.concat([son,veriler.iloc[:,1:3]],axis=1)
    windy=pd.DataFrame(data=windy,index=range(len(windy)),columns=['windy'])
    play=pd.DataFrame(data=play,index=range(len(play)),columns=['play'])
    
    last1=pd.concat([enson,windy],axis=1)
    last=pd.concat([last1,play],axis=1)
    
    print(windy)
    
    
    
    
    
    