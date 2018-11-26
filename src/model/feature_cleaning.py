import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#FIRST
def load_dfs() -> 'dfs':
    '''loads all the csvs and text needed'''

    df = pd.read_csv('~/galv/ACS-PUMS-data/csv_pca/psam_p06.csv')

    fieldofdegree_df = pd.read_csv('~/galv/capstone/resources/ACSPUMS2017CodeLists-FieldofDegree.csv', 
                            header=2, usecols=['2017 PUMS code', '2017 PUMS Field of Degree Description'])

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

    major_majors = pd.read_csv('~/galv/capstone/resources/FOD1P_maj_labels.csv', header=None, names=['code','major major'])

    NAICSP_labels_df = pd.read_csv('~/galv/capstone/resources/NAICScode.csv', header=None, names=['code','NAICSP_labels'])

    MAJ_NAICSP_labels_df = pd.read_csv('~/galv/capstone/resources/MAJ_NAICS_labels.csv', header=None, names=['code','MAJ_NAICSP_labels'])
    MAJ_NAICSP_labels_df.code = MAJ_NAICSP_labels_df.code.astype(str)

    return df, fieldofdegree_df, SOCP_labels, schl_labels, major_majors, NAICSP_labels_df, MAJ_NAICSP_labels_df


#SECOND
def clean_that_target(df, SOCP_labels) -> 'df':
    '''prepares the target variable SOCP for use in all subsestrquent dfs, and filters the rows for my desired sample'''

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


#SECOND PART 2 OPTIONAL
def single_occ_target(youngemp_df) -> 'df':
    '''single occ target for experimenting, run after clean that target'''
    mymap = {'15': 1}
    youngemp_df['MAJ_SOCP_15'] = youngemp_df.MAJ_SOCP.map(mymap).fillna(0).astype(int)
    return youngemp_df

#SECOND PART 2 ALTERNATIVE - exclusively computer occupations
def single_occ_target_specific(youngemp_df) -> 'df':
    '''single occ target for experimenting, run after clean that target'''
    mymap = {
    '151111':1,
    '151121':1,
    '151122':1,
    '151131':1,
    '15113X':1,
    '151134':1,
    '151141':1,
    '151142':1,
    '151143':1,
    '151150':1,
    '151199':1}
    youngemp_df['SOCP_computer'] = youngemp_df.SOCP.map(mymap).fillna(0).astype(int)
    return youngemp_df

# THIRD
def create_edu_df(youngemp_df, fieldofdegree_df, schl_labels, major_majors) -> 'df':
    '''df used to examine relationships between education and SOCP. 
    this is the initial feature subset. make this first. then the NAICSP_SOCP_df, or the full 16 features
    this contains redundant features'''

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
    edu_df = FOD2P_df[['SERIALNO','SOCP', 'MAJ_SOCP', 'MAJ_SOCP_labels', 'MAJ_SOCP_43', 'FOD1P', 
                        'FOD2P', 'FOD1P_labels', 'FOD2P_labels', 'SCHL']]

    #create column for SCHL label names
    edu_df.SCHL = edu_df.SCHL.astype(int).astype(str)
    edu_df['SCHL_labels'] = edu_df.SCHL.map(schl_labels)
    edu_df['SCHL_ord'] = edu_df.SCHL.astype(int)

    #make major majors
    edu_df['FOD1P_MAJ'] = edu_df['FOD1P'].str.slice(start=0, stop=2)
    major_majors.code = major_majors.code.str.slice(start=0, stop=2).astype(int) + 10
    major_majors.code = major_majors.code.astype(str)
    major_majors['major major'] = major_majors['major major'].str.lower().str.capitalize()
    #merge
    edu_df = edu_df.merge(major_majors, how='left', left_on='FOD1P_MAJ', right_on='code')
    edu_df.rename({'major major': 'FOD1P_MAJ_labels'}, axis=1, inplace=True)
    edu_df.drop(columns='code', inplace=True)
    edu_df['FOD1P_MAJ'] = edu_df['FOD1P_MAJ'].astype(int)

    print('before dummies:')
    print(edu_df.info(memory_usage='deep')) # check for no missing data

    dummies = pd.get_dummies(edu_df, columns=['SCHL_labels', 'FOD1P_labels', 'FOD2P_labels', 'FOD1P_MAJ_labels'], 
                            prefix=['SCHL_', 'FOD1P_', 'FOD2P_', 'FOD1P_MAJ_'], drop_first=False)
    #concat might work below but you must assign the merge to a new variable
    cols_to_use = dummies.columns.difference(edu_df.columns)
    edu_df2 = pd.merge(edu_df, dummies[cols_to_use], left_index=True, right_index=True,  validate='1:1',how='outer')

    return edu_df2





# THIRD ALTERNATE
def create_freewill_df(youngemp_df, fieldofdegree_df, schl_labels, major_majors) -> 'df':
    ''' df used to examine relationship between all 16 free will features and target'''
    
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
    edu_df = FOD2P_df
    
    #make major majors
    edu_df['FOD1P_MAJ'] = edu_df['FOD1P'].str.slice(start=0, stop=2)
    major_majors.code = major_majors.code.str.slice(start=0, stop=2).astype(int) + 10
    major_majors.code = major_majors.code.astype(str)
    major_majors['major major'] = major_majors['major major'].str.lower().str.capitalize()
    #merge
    edu_df = edu_df.merge(major_majors, how='left', left_on='FOD1P_MAJ', right_on='code')
    edu_df.rename({'major major': 'FOD1P_MAJ_labels'}, axis=1, inplace=True)
    edu_df.drop(columns='code', inplace=True)
    edu_df['FOD1P_MAJ'] = edu_df['FOD1P_MAJ'].astype(int)

     #create column for SCHL label names
    edu_df.SCHL = edu_df.SCHL.astype(int).astype(str)
    edu_df['SCHL_labels'] = edu_df.SCHL.map(schl_labels)
    edu_df['SCHL_ord'] = edu_df.SCHL.astype(int)


    #------------------------------------------after this part is new for freewill variables, until 'before dummies'
    freewill_df = edu_df[['SERIALNO','SOCP', 'MAJ_SOCP', 'MAJ_SOCP_labels', 'SOCP_computer', 'FOD1P', 
                        'FOD2P', 'FOD1P_labels', 'FOD1P_MAJ_labels', 'FOD2P_labels', 'SCHL', 'SCHL_labels', 'PUMA', 'COW', 
                        'ENG', 'JWTR', 'JWMNP', 'MARHT', 'WKHP', 'WKW', 'MSP', 'RELP', 'FMILSP']]
    #update the target as you change it

    #PUMA, incl as both int and cat?
    freewill_df['PUMA_cat'] = freewill_df.PUMA.astype(str)

    #COW
    COW_df = {
    1:'Employee(for-profit)',
    2:'Employee(non-profit)',
    3:'Local gov employee',
    4:'State govemployee',
    5:'Federal gov employee',
    6:'Self-employed(not inc)',
    7:'Self-employed(inc)',
    8:'Family business, unpaid'}
    freewill_df['COW_labels'] = freewill_df.COW.map(COW_df)

    #ENG  
    ENG_df = {
    9999:'Native',
    1:'Very well',
    2:'Well',
    3:'Not well',
    4:'Not at all'}
    freewill_df.ENG = freewill_df.ENG.fillna(9999)
    freewill_df['ENG_labels'] = freewill_df.ENG.map(ENG_df)

    #JWTR
    JWTR_df = {
    9999:'N/A',
    1:'Automobile',
    2:'Bus',
    3:'Streetcar',
    4:'Subway',
    5:'Railroad',
    6:'Ferryboat',
    7:'Taxicab',
    8:'Motorcycle',
    9:'Bicycle',
    10:'Walked',
    11:'Worked at home',
    12:'Other method'}
    freewill_df.JWTR = freewill_df.JWTR.fillna(9999)
    freewill_df['JWTR_labels'] = freewill_df.JWTR.map(JWTR_df)

    #JWMNP  first real numeric variable!!
    freewill_df.JWMNP = freewill_df.JWMNP.fillna(0)

    #MARHT
    MARHT_df = {
    9999: 'Never married',
    1:'Once',
    2:'Twice',
    3:'Thrice or more'}
    freewill_df.MARHT = freewill_df.MARHT.fillna(9999)
    freewill_df['MARHT_labels'] = freewill_df.MARHT.map(MARHT_df)

    #WKHP   numeric, working hours
    freewill_df.WKHP = freewill_df.WKHP.fillna(0)

    #WKW    working weeks past 12 months
    freewill_df.WKW = freewill_df.WKW.fillna(0)

    WKW_df ={
    0:'N/A',
    1:'50 to 52',
    2:'48 to 49',
    3:'40 to 47',
    4:'27 to 39',
    5:'14 to 26',
    6:'< 14'}
    freewill_df['WKW_labels'] = freewill_df.WKW.map(WKW_df)

    switch_WKW_df = {
    'N/A':1,
    '< 14':2,
    '14 to 26':3,
    '27 to 39':4,
    '40 to 47':5,
    '48 to 49':6,
    '50 to 52':7}
    freewill_df['WKW_ord'] = freewill_df.WKW_labels.map(switch_WKW_df)

    #MSP married spouse present - remember these will corr with age
    freewill_df.MSP.fillna(9999)
    MSP_df = {
    9999:'N/A (age less than 15 years)',
    1	:'Now married, spouse present',
    2	:'Now married, spouse absent',
    3	:'Widowed',
    4	:'Divorced',
    5	:'Separated',
    6	:'Never married'}
    freewill_df['MSP_labels'] = freewill_df.MSP.map(MSP_df)

    # #SFR  not populous
    # SFR_df = {
    # 9999:'N/A (GQ/not in a subfamily)',
    # 1   :'Husband/wife no children',
    # 2   :'Husband/wife with children',
    # 3   :'Parent in a one-parent subfamily',
    # 4   :'Child in a married-couple subfamily',
    # 5   :'Child in a mother-child subfamily',
    # 6   :'Child in a father-child subfamily'}

    #RELP
    RELP_df = {
    0:  'Reference person',
    1:  'Husband/wife',
    2:  'Biological son or daughter',
    3:  'Adopted son or daughter',
    4:  'Stepson or stepdaughter',
    5:  'Brother or sister',
    6:  'Father or mother',
    7:  'Grandchild',
    8:  'Parent-in-law',
    9:  'Son-in-law or daughter-in-law',
    10: 'Other relative',
    11: 'Roomer or boarder',
    12: 'Housemate or roommate',
    13: 'Unmarried partner',
    14: 'Foster child',
    15: 'Other nonrelative',
    16: 'Institutionalized group quarters population',
    17: 'Noninstitutionalized group quarters population',
    9999: 'N/A'}
    freewill_df.RELP.fillna(9999)
    freewill_df['RELP_labels'] = freewill_df.RELP.map(RELP_df)

    #FMILSP flag military
    FMILSP_df ={
        0:'No',
        1:'Yes'
    }
    freewill_df['FMILSP_labels'] = freewill_df.FMILSP.map(FMILSP_df)

    print('before dummies:')
    print(freewill_df.info(memory_usage='deep')) # check for no missing data

    dummies = pd.get_dummies(freewill_df, 
            columns=['SCHL_labels', 
                    'FOD1P_labels', 
                    'FOD2P_labels', 
                    'FOD1P_MAJ_labels',
                    'PUMA_cat',
                    'COW_labels',
                    'ENG_labels',
                    'JWTR',
                    'MARHT_labels',
                    'MSP_labels',
                    'RELP_labels',
                    'FMILSP_labels'], 
                    prefix=['SCHL_', 
                            'FOD1P_', 
                            'FOD2P_', 
                            'FOD1P_MAJ_', 
                            'PUMA_',
                            'COW_',
                            'ENG_',
                            'JWTR_',
                            'MARHT_',
                            'MSP_',
                            'RELP_',
                            'FMILSP_'], drop_first=False)
    #concat might work below but you must assign the merge to a new variable
    cols_to_use = dummies.columns.difference(freewill_df.columns)
    freewill_df2 = pd.merge(freewill_df, dummies[cols_to_use], left_index=True, right_index=True,  validate='1:1',how='outer')

    freewill_df2.drop(columns=['FOD1P', 'FOD2P', 'COW', 'ENG', 'JWTR', 'MARHT', 'WKW', 'MSP', 'RELP', 'FMILSP', 'PUMA_cat'], inplace=True)
    
    return freewill_df2








# this is used to cluster the target later
def create_NAICSP_SOCP_df(youngemp_df, NAICSP_labels_df, MAJ_NAICSP_labels_df) -> 'df':
    '''df used to examine relationships between NAICS and SOCP, for clustering a new target'''

    NAICSP_SOCP_df = youngemp_df[['SERIALNO', 'SOCP', 'MAJ_SOCP', 'MAJ_SOCP_labels', 'NAICSP']]
    #merge naicsp labels
    NAICSP_SOCP_df = NAICSP_SOCP_df.merge(NAICSP_labels_df, how='left', left_on='NAICSP', right_on='code')
    NAICSP_SOCP_df.drop(columns=['code'], inplace=True)
    NAICSP_SOCP_df.NAICSP_labels = NAICSP_SOCP_df.NAICSP_labels.fillna(9999)
    NAICSP_SOCP_df['MAJ_NAICSP'] = NAICSP_SOCP_df.NAICSP.str.slice(start=0, stop=2)
    print(NAICSP_SOCP_df.info(memory_usage='deep')) # check for no missing data
    NAICSP_SOCP_df.MAJ_NAICSP = NAICSP_SOCP_df.MAJ_NAICSP.fillna(999)
    print(NAICSP_SOCP_df.info(memory_usage='deep')) # check for no missing data
    
    #merge maj naicsp labels
    NAICSP_SOCP_df = NAICSP_SOCP_df.merge(MAJ_NAICSP_labels_df, how='left', left_on='MAJ_NAICSP', right_on='code')
    NAICSP_SOCP_df.drop(columns=['code'], inplace=True)

    #make dummies
    dummies = pd.get_dummies(NAICSP_SOCP_df, columns=['MAJ_SOCP_labels', 'MAJ_NAICSP_labels'], prefix=['MAJ_SOCP_', 'MAJ_NAICSP_'], drop_first=False)
    NAICSP_SOCP_df = pd.concat([NAICSP_SOCP_df, dummies], axis=1) # why is this duplicating my columns??
    #NAICSP_SOCP_df.drop(columns=['SERIALNO', 'SOCP', 'MAJ_SOCP', 'NAICSP', 'NAICSP_labels', 'MAJ_NAICSP']) #bugfix for dups

    return NAICSP_SOCP_df







# #if __name__ == "__main__":
#     df, fieldofdegree_df, SOCP_labels, schl_labels = load_dfs()
#     youngemp_df = clean_that_target(df, SOCP_labels)


