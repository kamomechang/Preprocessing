# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import plotting 
from sklearn.preprocessing import StandardScaler

def main(df_data):
    scaler = StandardScaler()
    scaler.fit(df_data)
    scaler.transform(df_data)
    dfs = pd.DataFrame(scaler.transform(df_data), columns=df_data.columns)
    
    dfs.to_csv("standardized_data.csv")
    
    return dfs


if __name__=="__main__":
    
    df_data=pd.read_csv("II_Imputed_data.csv", index_col=0, sep=",")
    
    main(df_data)
