import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix, fbeta_score
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
import gc as gc
from sklearn.feature_selection import RFE


#-------------------------------------------------------------------------------------------------------------------------
def kfold_smote_RFE(features_num, classifier, folds, df_train_filtered_std, y_train, smote='y'):
    """K_fold training/validation for RFE with LightGBM/RandomForest/XGBoost/CATBoost,
    with SMOTE train re-sampling,
    features_num-> select the number of features for RFE"""

    # get a list of models to evaluate
    def get_models():
        models = dict()
        for i in range(2, features_num+1):
            models[str(i)] = RFE(estimator=classifier, n_features_to_select=i)
        return models

    # data from each foldf
    fold_results = list()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train_filtered_std, y_train)):
        train_x, train_y = df_train_filtered_std.iloc[train_idx], y_train.iloc[train_idx]
        valid_x, valid_y = df_train_filtered_std.iloc[valid_idx], y_train.iloc[valid_idx]

        # summarize class distribution
        counter = Counter(train_y)
        print('\n-----------------------------------------------------')
        print('Fold %2d, original distribution: ' % (n_fold + 1))
        print(counter)

        if smote=='y':
            # transform the dataset
            oversample = BorderlineSMOTE()
            train_x, train_y = oversample.fit_resample(train_x, train_y)
            # summarize the new class distribution
            counter = Counter(train_y)
            print('Fold %2d, re-sampled distribution: ' % (n_fold + 1))
            print(counter)


        # get the models to evaluate
        models = get_models()

        # evaluate the models and store results
        models_results, names = list(), list()
        for name, model in models.items():

            # Print the number of features of the model
            print('\nFeatures:%s' % (name))

            # fit RFE
            model.fit(train_x, train_y)

            # validation per model
            probas = model.predict_proba(valid_x)[:, 1]

            # ROC-AUC per model
            AUC = roc_auc_score(valid_y, probas)

            # Collecting results
            models_results.append(AUC)
            names.append(name)

            # summarize all features
            for i in range(train_x.shape[1]):
                print('Column: %d, Selected %s, Rank: %.3f' % (i, model.support_[i], model.ranking_[i]))

            # Print AUC score
            print(f'\nAUC: {AUC}')

        print('\nModels results')
        print(models_results)

        fold_results.append(models_results)

    print('\nFolds results')
    print(fold_results)

    fold_results = np.asarray(fold_results)
    # plot model performance for comparison
    plt.figure(figsize=(15,10))
    plt.boxplot(fold_results, labels=range(2,features_num+1), showmeans=True)
    plt.title('RECURSIVE FEATURE ELIMINATION'
             f'\n\ntrain re-sampling (SMOTE):"{smote}"',fontsize=20)
    plt.xlabel('Numbers of features selected',fontsize=15)
    plt.ylabel('Crossvalidation AUC',fontsize=15)
    plt.ylim((0.5, 0.8))
    
    # save
    plt.savefig(f'projets\\07_loan_customer_scoring\\production\\savefig\\model_test_{smote_case}\\feature_selection\\{class_weigh_case}\\feature_selection_RFE_feature_number.png', transparent=True)
    
    plt.show()

    return fold_results

#-------------------------------------------------------------------------------------------------------------------------
# Classification with kfold available for several algorithms
def kfold_classif(classifier, folds, df_train_std, target_train, df_val_std, target_val, custom_loss, fbeta, fbeta_number=0, logistic_regression=False, train_resampling='n', eval_set=False, scorer='auc', early_stopping_rounds=None, verbose=200):

    """K_fold training/validation for DecisionTree/RandomForest/LightGBM/XGBoost/CATBoost/LogisticRegression,
    train_resampling-> borderline smote re-sampling on the train part,
    fbetanumber-> for function to optimize"""

    """"num_iteration=clf.best_iteration_    can be added in the predict_proba() when callable """
    
# Create arrays and dataframes to store results
    crossvalid_probas = np.zeros(df_train_std.shape[0])
    valid_probas = np.zeros(df_val_std.shape[0])
    fold_AUC_list = []
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train_std.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    # Modification of columns
    df_train_std_2 = df_train_std[feats]
    df_val_std_2 = df_val_std[feats]
    df_train_std_2.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_train_std_2.columns]
    df_val_std_2.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_val_std_2.columns]

    # define thresholds
    thresholds = np.arange(0, 1, 0.001)
    # apply threshold to positive probabilities to create labels
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')
    
    def custom_cost_function(testy, yhat):
        # get the fn and the fp from the confusion matrix
        tn, fp, fn, tp = confusion_matrix(testy, yhat).ravel()
        # function
        y = 10*fn + fp
        return y
        
    # data from each fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train_std_2, target_train)):
        train_x, train_y = df_train_std_2.iloc[train_idx], target_train.iloc[train_idx]
        valid_x, valid_y = df_train_std_2.iloc[valid_idx], target_train.iloc[valid_idx]   
        
        # Re-sampling 
        if train_resampling=='y':
            # summarize class distribution
            counter = Counter(train_y)
            print('Fold %2d, original distribution: ' % (n_fold + 1))
            print(counter)
            
            # transform the dataset
            oversample = BorderlineSMOTE()
            train_x, train_y = oversample.fit_resample(train_x, train_y)
            # summarize the new class distribution
            counter = Counter(train_y)
            print('Fold %2d, re-sampled distribution: ' % (n_fold + 1))
            print(counter)
     
        # classifier instance
        clf = classifier

        # fitting
        if eval_set==True:
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric=scorer, verbose=verbose, early_stopping_rounds=early_stopping_rounds)
        
        if eval_set==False:
            clf.fit(train_x, train_y)

        # validation
        crossvalid_probas[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        
        # ROC-AUC
        AUC = roc_auc_score(valid_y, crossvalid_probas[valid_idx])
        fold_AUC_list.append(AUC)

        # showing results from each fold
        print('Fold %2d AUC : %.6f' % (n_fold + 1, AUC))
           
        # Collecting results
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        
        # Classifier case
        if logistic_regression==True:
            fold_importance_df["importance"] = clf.coef_[0]
        if logistic_regression==False:
            fold_importance_df["importance"] = clf.feature_importances_
            
        fold_importance_df["fold"] = n_fold + 1
        fold_importance_df["val_fold_AUC"] = AUC
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
                
        #validation_ROC_AUC = roc_auc_score(target_train, crossvalid_probas)
        valid_probas += clf.predict_proba(df_val_std)[:, 1] / folds.n_splits      
                
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    # Final performance
    mean_crossvalid_fold_ROC_AUC = sum(fold_AUC_list)/len(fold_AUC_list)
    print('Mean cross-validation ROC-AUC score %.6f' % mean_crossvalid_fold_ROC_AUC)
    
    #validation_ROC_AUC = roc_auc_score(target_train, crossvalid_probas)
    validation_ROC_AUC = roc_auc_score(target_val, valid_probas)
    print('Validation ROC-AUC score %.6f' % validation_ROC_AUC)
    
    # Optimising the threshold  
    
    if (fbeta==True)&(fbeta_number!=0):
        # evaluate each threshold with f-beta loss function
        scores = [fbeta_score(target_val.values, to_labels(valid_probas, t), average='weighted', beta=fbeta_number) for t in thresholds]
    
        # get best threshold
        ix = np.argmax(scores)
        print(f'Threshold=%.3f, F-{fbeta_number} score_max=%.5f' % (thresholds[ix], scores[ix]))
        best_score = scores[ix]
        threshold = thresholds[ix]
        
        
    if custom_loss=='y':
        # evaluate each threshold with custom loss function
        scores = [custom_cost_function(target_val.values, to_labels(valid_probas, t)) for t in thresholds]
        
        # get best threshold
        ix = np.argmin(scores)
        print(f'Threshold=%.3f, Custom loss function (10*fn + fp) score_min=%.5f' % (thresholds[ix], scores[ix]))
        best_score = scores[ix]
        threshold = thresholds[ix]
        
        
    return clf, feature_importance_df, mean_crossvalid_fold_ROC_AUC, validation_ROC_AUC, best_score, threshold
    
    
#-------------------------------------------------------------------------------------------------------------------------
# One hot encoder (avec récupération des labels)

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import numpy as np

class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_<{self.categories_[i][j]}>')
                j += 1
        return new_columns
    
#-------------------------------------------------------------------------------------------------------------------------

# Targer Encoding ou One Hot Encoding (1 nouvelle colonne crée)
def encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]], columns=[column,fix_column])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = trained_model.transform(dataframe_work[column])
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe


# Label Encoding ou One Hot Encoding (1 nouvelle colonne crée)

def label_encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]], columns=[column,fix_column])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = dataframe_work[column].apply(lambda x: trained_model.transform([x])[0] if pd.notna(x) else np.NaN)
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe

# Targer Encoding ou One Hot Encoding (1 nouvelle colonne crée)
def target_encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]], columns=[column,fix_column])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = trained_model.transform(dataframe_work[column])
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe


# ONE-HOT-ENCODING (plusieurs nouvelles colonnes crées)
def vector_encoding_transform_with_merge(dataframe, column, fix_column, trained_model):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work_transformed = pd.DataFrame(trained_model.transform(dataframe_work))

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work_transformed.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work_transformed, on=fix_column)
    
    return dataframe

#----------------------------------------------------------------------------------------------

def SAVE_encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = trained_model.transform(dataframe_work[column])
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe




