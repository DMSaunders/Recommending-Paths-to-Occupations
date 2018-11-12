import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('~/galv/ACS-PUMS-data/csv_pca/psam_p06.csv')

SOCP_labels = {'11': 'Management', 
               '13': 'Business and Financial Operations', 
               '15': 'Computer and Mathematical', 
               '17': 'Architecture and Engineering', 
               '19': 'Life, Physical, and Social Science', 
               '21': 'Community and Social Service', 
               '23': 'Legal', 
               '25': 'Education, Training, and Library', 
               '27': 'Arts, Design, Entertainment, Sports, and Media', 
               '29': 'Healthcare Practitioners and Technical', 
               '31': 'Healthcare Support',
               '33': 'Protective Service',
               '35': 'Food Preparation and Serving Related',
               '37': 'Building and Grounds Cleaning and Maintenance',
               '39': 'Personal Care and Service',
               '41': 'Sales and Related',
               '43': 'Office and Administrative Support',
               '45': 'Farming, Fishing, and Forestry',
               '47': 'Construction and Extraction',
               '49': 'Installation, Maintenance, and Repair',
               '51': 'Production',
               '53': 'Transportation and Material Moving',
               '55': 'Military Specific'}

def clean_that_data(df):

    #new df with nan in SOCP dropped
    SOCPdf = df.dropna(axis='index', subset=['SOCP'])[df.SOCP != '999920']

    # make a new feature for major SOCP category
    SOCPdf['MAJ_SOCP'] = SOCPdf.SOCP.str.slice(start=0, stop=2)

    print('Number of employed people: {}'.format(SOCPdf.SOCP.count()))

    print('Percent employed people: {}'.format(SOCPdf.SOCP.count()/df.SERIALNO.count())) 
    
    #create column for label names
    SOCPdf['MAJ_SOCP_labels'] = SOCPdf.MAJ_SOCP.map(SOCP_labels)

    #create new df for people under 35
    youngemp_df = SOCPdf[SOCPdf.AGEP <= 35]

    