#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 20:52:02 2023

@author: isa
"""

"""
Loading packages
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
"""
Setting working directory 
"""
os.chdir("/Users/isa/Downloads/pythondataMLU220213600/220213600/homework/02_hometask")

"""
First group of Tasks 
"""
#Importing dataset
prices = pd.read_csv('02_python_data.csv')
returns = prices.pct_change()
logreturns = np.log(prices) - np.log(prices.shift(1))
display(returns.head)