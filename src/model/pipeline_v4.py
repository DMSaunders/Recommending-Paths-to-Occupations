import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

# import sys 
# sys.path.append('../')
#import feature_cleaning as feature_cleaning

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#sklearn models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#sklearn other
#import graphviz 
from sklearn.model_selection import train_test_split, GridSearchCV
<<<<<<< HEAD
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, accuracy_score
=======
#from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __name__ == "__main__":

#     df, fieldofdegree_df, SOCP_labels, schl_labels, major_majors, NAICSP_labels_df, MAJ_NAICSP_labels_df = feature_cleaning.load_dfs()

#     youngemp_df = feature_cleaning.clean_that_target(df, SOCP_labels)
#     youngemp_df = feature_cleaning.single_occ_target(youngemp_df)
    edu_df = pd.read_csv('edu_df_15.csv')

    # split the data, choosing only edu cols
    X = edu_df.drop(columns=[ 'Unnamed: 0', 'SERIALNO', 'FOD1P', 'FOD2P','SOCP','MAJ_SOCP','MAJ_SOCP_labels', 
                    'MAJ_SOCP_15','FOD1P_labels','FOD2P_labels','SCHL',
                    'SCHL_labels','FOD1P_MAJ_labels', 'FOD1P_MAJ'])
    y = edu_df.loc[:,'MAJ_SOCP_15']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3, 
                                                        random_state=42)


    # model pipelines
    #-----------------------------------
    #-------------linear
    pipe_lr = Pipeline([('scl', StandardScaler()),
                ('clf', LogisticRegression(random_state=42))])

    pipe_lr_l2 = Pipeline([('scl', StandardScaler()),
                ('clf', LogisticRegression(random_state=42))])

    pipe_sgd = Pipeline([('scl', StandardScaler()),
                ('clf', SGDClassifier(random_state=42))])


    #-------------trees
    pipe_dt = Pipeline([('clf', DecisionTreeClassifier(random_state=42))])

    pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=42))])

    pipe_rf_scl = Pipeline([('scl', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=42))])

    pipe_gb = Pipeline([('clf', GradientBoostingClassifier(random_state=42))])


    #-------------SVM
    pipe_svm = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=42))])


    #-------------KNN
    pipe_knn = Pipeline([('clf', KNeighborsClassifier())])

    pipe_knn_scl = Pipeline([('scl', StandardScaler()),
                ('clf', KNeighborsClassifier())])

    #-----------------------------------


    # grid search params
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_range_fl = [1.0, 0.5, 0.1]
    max_depth = [10,100,1000,10000]
    alpha_range = [.1, .001, .00001, .000001]
    gamma_range = [.1, 1, 10]

    #-------------linear
    grid_params_lr = [{'clf__penalty': ['l1'],
            'clf__C': param_range_fl,
            'clf__solver': ['liblinear', ],  #,'saga'
            #'clf__multi_class': ['ovr', 'multinomial', 'auto'],
            'clf__class_weight': [None, 'balanced']}] 

    grid_params_lr_l2 = [{'clf__penalty': ['l2'],
            'clf__C': param_range_fl,
            'clf__solver': ['newton-cg', 'lbfgs', 'liblinear'],  #, 'sag'
            #'clf__multi_class': ['ovr', 'multinomial', 'auto'],
            'clf__class_weight': [None, 'balanced']}]

    grid_params_sgd = [{'clf__loss': ['hinge', 'log', 'perceptron'],
            'clf__alpha': alpha_range,
            'clf__penalty': ['l1', 'l2', 'elasticnet'],
            'clf__class_weight': [None, 'balanced']}] 

    #-------------trees
    grid_params_dt = [{'clf__criterion': ['gini', 'entropy'],
            'clf__min_samples_leaf': param_range,
            'clf__max_depth': max_depth,
            'clf__min_samples_split': param_range[1:],
            'clf__class_weight': [None, 'balanced']}]

    grid_params_rf = [{'clf__criterion': 'entropy',
            'clf__min_samples_leaf': 5,
            'clf__max_depth': 100,,
            'clf__min_samples_split': 10,
            'clf__class_weight': None}]

    grid_params_gb = [{'clf__loss': ['deviance', 'exponential'],
            'clf__learning_rate': alpha_range,
            'clf__n_estimators': max_depth,
            'clf__subsample': param_range_fl}]

    #-------------SVM
    grid_params_svm = [{'clf__kernel': ['linear', 'rbf', 'poly'],
            'clf__degree': param_range[1:],
            'clf__gamma': gamma_range,
            'clf__C': gamma_range,
            'clf__class_weight': [None, 'balanced']}]

    #-------------KNN
    grid_params_knn = [{'clf__n_neighbors': param_range}]

    #--------------------------------------------------------------

    # Construct grid searches
    jobs = -1
    verbose = 1

    #-------------linear
    gs_lr = GridSearchCV(estimator=pipe_lr,
                param_grid=grid_params_lr,
                scoring='f1_micro',
                cv=10,
                n_jobs=jobs,
                verbose=verbose) 

    gs_lr_l2 = GridSearchCV(estimator=pipe_lr_l2,
                param_grid=grid_params_lr_l2,
                scoring='f1_micro',
                cv=10,
                n_jobs=jobs,
                verbose=verbose)

    gs_sgd = GridSearchCV(estimator=pipe_sgd,
                param_grid=grid_params_sgd,
                scoring='f1_micro',
                cv=10,
                verbose=verbose)

        
    #-------------trees    
    gs_dt = GridSearchCV(estimator=pipe_rf,
                param_grid=grid_params_dt,
                scoring='f1_micro',
                cv=10, 
                n_jobs=jobs,
                verbose=verbose)

    gs_rf = GridSearchCV(estimator=pipe_rf,
                param_grid=grid_params_rf,
                scoring='f1_micro',
                cv=10, 
                n_jobs=jobs,
                verbose=verbose)

    gs_rf_scl = GridSearchCV(estimator=pipe_rf_scl,
                param_grid=grid_params_rf,
                scoring='f1_micro',
                cv=10, 
                n_jobs=jobs,
                verbose=verbose)

    gs_gb = GridSearchCV(estimator=pipe_gb,
                param_grid=grid_params_gb,
                scoring='f1_micro',
                cv=10, 
                verbose=verbose)

    #-------------SVM

    gs_svm = GridSearchCV(estimator=pipe_svm,
                param_grid=grid_params_svm,
                scoring='f1_micro',
                cv=10,
                n_jobs=jobs,
                verbose=verbose)

    #-------------KNN
    gs_knn = GridSearchCV(estimator=pipe_knn,
                param_grid=grid_params_knn,
                scoring='f1_micro',
                cv=10,
                n_jobs=jobs,
                verbose=verbose)

    gs_knn_scl = GridSearchCV(estimator=pipe_knn_scl,
                param_grid=grid_params_knn,
                scoring='f1_micro',
                cv=10,
                n_jobs=jobs,
                verbose=verbose)

    #---------------------------------------------------------------------

    # List of pipelines for ease of iteration
    grids = [ gs_rf]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0:'rf'}

    # Fit the grid search objects
    print('Performing model optimizations...')
    best_f1_micro = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        gs.fit(X_train, y_train)
        
        # Best params
        print('Best params: %s' % gs.best_params_)
        
        # Best training data f1
        print('Best training f1: %.3f' % gs.best_score_)
        
        # Predict on test data with best params
        y_pred = gs.predict(X_test)

        #accuracy
        print('test accuracy {}:'.format( accuracy_score(y_train, y_pred)))
        
        # Test data accuracy of model with best params
        print('Test set f1 score for best params: %.3f ' % f1_score(y_test, y_pred))
        
        # Track best (highest test f1) model
        if f1_score(y_test, y_pred) > best_f1_micro:
            best_f1_micro = f1_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best test set f1: %s' % grid_dict[best_clf])

    # Save best grid search pipeline to file
    dump_file = 'best_model_no_feat_sel_extr_occ_15.pkl'
    joblib.dump(best_gs, dump_file, compress=1)
    print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))
