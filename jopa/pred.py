import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import os

train_File = 'train.csv'


data = pd.read_csv(train_File)