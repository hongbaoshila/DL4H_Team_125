#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: Canruo Zou
"""

import pandas as pd
import io
import os
import numpy as np

def label(row):
   if row['ORG_ITEMID'] == 80002:
      return 1
   return 0

def main():
  
# reading two csv files 
    data1 = pd.read_csv('./new.csv') 
    data2 = pd.read_csv('./new1.csv') 
  
# using merge function by setting how='inner' 
    output1 = pd.merge(data1, data2,  
                   on='SUBJECT_ID',  
                   how='inner')
    
    output1['Label'] = output1.apply(label, axis=1)
    df1 = output1[['HADM_ID','ROW_ID','CHARTDATE','CHARTTIME','TEXT','Label']]
    df1.to_csv('./newdata.csv', index=False)


if __name__ == "__main__":
    main()