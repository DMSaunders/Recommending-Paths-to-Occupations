import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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

def clean_that_target(df) -> 'df':

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

    print('Number of young employed people: {}'.format(youngemp_df.SOCP.count()))

    print('Percent young employed people(out of all PUMS): {}'.format(youngemp_df.SOCP.count()/df.SERIALNO.count()))

    print('Number of emp cats: {}'.format(youngemp_df.MAJ_SOCP.value_counts().count()))

    return youngemp_df

def create_edu_df(df) -> 'df':

    print('Number of degree fields present (max 173): {}'.format(youngemp_df.FOD1P.value_counts().count()))

    #fill missing major codes, format
    youngemp_df.FOD1P = youngemp_df.FOD1P.fillna(9999)
    youngemp_df.FOD1P = youngemp_df.FOD1P.astype(int).astype(str)
    youngemp_df.FOD2P = youngemp_df.FOD2P.fillna(9999)
    youngemp_df.FOD2P = youngemp_df.FOD2P.astype(int).astype(str)

    #format degree labels df, create col
    fieldofdegree_df.dropna(axis='rows', how='any', inplace=True)
    fieldofdegree_df['2017 PUMS code'] = fieldofdegree_df['2017 PUMS code'].astype(int).astype(str)
    
    #merge degree labels and youngemp, format
    FOD1P_df = youngemp_df.merge(fieldofdegree_df, how='left', left_on='FOD1P', right_on='2017 PUMS code')
    #format FOD1P_labels
    FOD1P_df.rename({'2017 PUMS Field of Degree Description': 'FOD1P_labels'}, axis=1, inplace=True)
    FOD1P_df.drop(columns='2017 PUMS code', inplace=True)
    FOD1P_df.FOD1P_labels.fillna('No major', inplace=True)
    
    #merge degree labels and FOD1P_df
    FOD2P_df = FOD1P_df.merge(fieldofdegree_df, how='left', left_on='FOD2P', right_on='2017 PUMS code')
    #format FOD2P_labels
    FOD2P_df.rename({'2017 PUMS Field of Degree Description': 'FOD2P_labels'}, axis=1, inplace=True)
    FOD2P_df.drop(columns='2017 PUMS code', inplace=True)
    FOD2P_df.FOD2P_labels.fillna('No major', inplace=True)

    #create edu_df limited to edu features
    edu_df = FOD2P_df[['SERIALNO','SOCP', 'MAJ_SOCP', 'MAJ_SOCP_labels', 'FOD1P', 'FOD2P', 'FOD1P_labels', 'FOD2P_labels', 'SCHL']]

    schl_labels = {'1':'No schooling completed',
                    '2':"Nursery school, preschool",
                    '3':"Kindergarten",
                    '4':"Grade 1",
                    '5':"Grade 2",
                    '6':"Grade 3",
                    '7':"Grade 4",
                    '8':"Grade 5",
                    '9':"Grade 6",
                    '10':"Grade 7",
                    '11':"Grade 8",
                    '12':"Grade 9",
                    '13':"Grade 10",
                    '14':"Grade 11",
                    '15':"12th grade - no diploma",
                    '16':"Regular high school diploma",
                    '17':"GED or alternative credential",
                    '18':"Some college, but less than 1 year",
                    '19':"1 or more years of college credit, no degree",
                    '20':"Associate's degree",
                    '21':"Bachelor's degree",
                    '22':"Master's degree",
                    '23':"Professional degree beyond a bachelor's degree",
                    '24':"Doctorate degree"}
    
    #create column for SCHL label names
    edu_df.SCHL = edu_df.SCHL.astype(int).astype(str)
    edu_df['SCHL_labels'] = edu_df.SCHL.map(schl_labels)


if __name__ == "__main__":
    #load dfs
    df = pd.read_csv('~/galv/ACS-PUMS-data/csv_pca/psam_p06.csv')
    
    fieldofdegree_df = pd.read_csv('~/galv/capstone/resources/ACSPUMS2017CodeLists-FieldofDegree.csv', 
                               header=2, usecols=['2017 PUMS code', '2017 PUMS Field of Degree Description'])