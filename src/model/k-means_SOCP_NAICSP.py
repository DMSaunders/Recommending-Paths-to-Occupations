import pandas as pd
import numpy as np
import sys 

#my funcs
sys.path.append('../')
import src.features.feature_cleaning as feature_cleaning

#sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from mlxtend.plotting import plot_confusion_matrix

#cleaning
df, fieldofdegree_df, SOCP_labels, schl_labels, major_major, NAICSP_labels_df, MAJ_NAICSP_labels = feature_cleaning.load_dfs()
youngemp_df = feature_cleaning.clean_that_target(df, SOCP_labels)
NAICSP_SOCP_df = feature_cleaning.create_NAICSP_SOCP_df(youngemp_df, NAICSP_labels_df, MAJ_NAICSP_labels)





KMeans(verbose=5, n_jobs=-1)