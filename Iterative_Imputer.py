# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

warnings.resetwarnings()
warnings.simplefilter('ignore')

def main(df_data):

    df_data.iloc[:, :]=IterativeImputer().fit_transform(df_data)
    df_data.to_csv("II_Imputed_data.csv")
    
    return df_data

if __name__=="__main__":
    
    df_data=pd.read_csv("ohe_encoded_data.csv", index_col=0, sep=",")
    
    df_data=main(df_data)
