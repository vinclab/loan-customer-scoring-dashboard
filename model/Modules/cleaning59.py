#BANQUE DE FONCTIONS DE NETTOYAGE DE DATAFRAME - Vincent Salas

# import des librairies dont nous aurons besoin
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------------------------------------------------
#NaN DATAFRAME
#-------------------------------------------------------------------------------------------------------------------------
# Définition d"une fonction identifiant les lignes sans données d'un dataframe
def na_rows_list(dataframe,value):
    """Fonction faisant la somme des éléments de ligne et retournant une liste d'indices de lignes sans données"""
    """La valeur de vérification conditionnelle dépend d'un projet, A CHANGER!!!!!!!!"""
    na_row_list = []
    for i in dataframe.index:
        if dataframe.loc[i].value_counts().sum() == value: # Attention valeur de condition à changer
            na_row_list.append(i)
            
    return na_row_list

# Définition d"une fonction supprimant les lignes sans données d'un dataframe 
def na_raw_drop_df(dataframe,liste):
    """Supprime les lignes NaN"""
    """Utilisation de la fonction na_raws au préalable pour trouver une liste de lignes à supprimer"""
    """Renvoie un dataframe"""
    for i in liste: # Utilise na_raws()
        dataframe.drop(i,inplace=True)
        
    return dataframe

#-------------------------------------------------------------------------------------------------------------------------
#LIGNES DATAFRAME
#-------------------------------------------------------------------------------------------------------------------------
# Taux de remplissage moyen par ligne
#def row_data_rate_mean(dataframe):
#    """Calcul du taux moyen de remplissage par ligne"""
#    rows_rate = []
#    for i in dataframe.index:
#        rate = dataframe.loc[i].value_counts().sum()/dataframe.shape[1]
#        rows_rate.append(rate)
#    
#    return sum(rows_rate)/len(rows_rate)

# Définition d"une fonction identifiant les lignes d'un dataframe avec un taux de remplissage minimal 
def min_row_data_rate_list(dataframe,value):
    """Fonction faisant la somme des éléments de ligne et retournant une liste d'indices de lignes avec un taux de  
    remplissage minimal"""
    """La valeur de vérification conditionnelle dépend d'un projet, A CHANGER!!!!!!!!"""
    rows_list = []
    for i in dataframe.index:
        if dataframe.loc[i].value_counts().sum()/dataframe.shape[1] > value:
            rows_list.append(i)
            
    return rows_list

# Définition d"une fonction supprimant les lignes d'un dataframe avec un taux de remplissage insuffisant
def min_row_data_rate_df(dataframe,value):
    """Fonction faisant la somme des éléments de ligne et retournant un dataframe avec un taux de  
    remplissage minimal"""
    """La valeur de vérification conditionnelle dépend d'un projet, A CHANGER!!!!!!!!"""
    rows_list = []
    for i in dataframe.index:
        if dataframe.loc[i].value_counts().sum()/dataframe.shape[1] < value:
            dataframe.drop(i,inplace=True)
            
    return dataframe

# Définition d"une fonction identifiant les lignes avec peu de données d'un dataframe
#def few_data_rows_list(dataframe,value):
#    """Fonction faisant la somme des éléments de ligne et retournant une liste d'indices de lignes avec peu de
#    données, remplissant une condition de remplissage < x données"""
#    """La valeur de vérification conditionnelle dépend d'un projet, A CHANGER!!!!!!!!"""
#    rows_list = []
#    for i in dataframe.index:
#        if dataframe.loc[i].value_counts().sum() < value:
#            rows_list.append(i)
#    return rows_list

# Définition d"une fonction identifiant les lignes avec un minimum de données d'un dataframe
#def enough_data_rows_list(dataframe,value):
#    """Fonction faisant la somme des éléments de ligne et retournant une liste d'indices de lignes avec assez de
#    données, remplissant une condition de remplissage > x données"""
#    """La valeur de vérification conditionnelle dépend d'un projet, A CHANGER!!!!!!!!"""
#    rows_list = []
#    for i in dataframe.index:
#        if dataframe.loc[i].value_counts().sum() > value:
#            rows_list.append(i)
#    return rows_list



#------------------------------------------------------------------------------------------------------------------------
#COLONNES DATAFRAME
#-------------------------------------------------------------------------------------------------------------------------
# Vérification du taux de remplissage par colonne
def column_data_rate(dataframe):
    """Fonction vérifiant le taux de remplissage par colonne"""
    serie_na = dataframe.notna().sum()/len(dataframe)
    
    return serie_na

# Taux de remplissage moyen par colonne:
def column_data_rate_mean(dataframe):
    """Calcul du taux moyen de remplissage par colonne"""
    serie_na = dataframe.notna().sum()/len(dataframe)
    
    return serie_na[:].mean()

# Liste des colonnes d'un dataframe à supprimer si non dans une liste comparative
def columns_not_in_list(dataframe, liste_garder):
    """Compare le nom des colonnes d'un dataframe avec une liste. Renvoie une liste de colonnes à supprimer si non dans la 
    liste voulue"""
    colonnes_supprimer = []
    for colonne in dataframe.columns:
        if colonne not in liste_garder:
            colonnes_supprimer.append(colonne)
            
    return colonnes_supprimer

# Suppression de colonnes d'un dataframe à partir d'une liste
def columns_delete_df(dataframe, list_delete):
    """Supprime les colonnes d'un dataframe non incluses dans une liste et retourne le dataframe"""
    dataframe.drop(list_delete,axis=1,inplace= True)
    
    return dataframe

# Définition d"une fonction supprimant les colonnes d'un dataframe avec un taux de remplissage insuffisant
def min_column_data_rate_df(dataframe,value):
    """Fonction retournant un dataframe avec un taux de remplissage minimal par colonne"""
    """La valeur de vérification conditionnelle dépend d'un projet, A CHANGER!!!!!!!!"""
    column_list = []
    for c in dataframe.columns:
        if dataframe[c].value_counts().sum()/dataframe.shape[0] < value:
            del dataframe[c]
            
    return dataframe

#-------------------------------------------------------------------------------------------------------------------------
# Valeurs aberrantes 
#-------------------------------------------------------------------------------------------------------------------------

# Calcul du nombre de valeurs aberrantes d'un dataframe
#def low_outlier_count(dataframe, columns, value):
#    """Calcul du nombre de valeurs aberrantes d'un dataframe en-dessous d'une valeur,  #impression du 
#    résultat"""
#    import numpy as np
#    dic_count_before = {}
#    dic_count_after = {}
#    dic_count_variables_aberrantes = {}
#
#    print('Nombre de valeurs aberrantes :\n')
#    for variable in columns:
#        dic_count_before[variable] = dataframe[variable].value_counts().sum()
#        dataframe[variable] = [t if t>value else np.NaN for t in dataframe[variable]]
#        dic_count_after[variable] = dataframe[variable].value_counts().sum()
#        dic_count_variables_aberrantes[variable] = dic_count_before[variable] - #dic_count_after[variable]
#        
#    return dic_count_variables_aberrantes

# Calcul du nombre de valeurs aberrantes d'un dataframe
#def high_outlier_count(dataframe, columns, value):
#    """Calcul du nombre de valeurs aberrantes d'un dataframe au-dessus d'une valeur,  #impression du 
#    résultat"""
#    import numpy as np
#    dic_count_before = {}
#    dic_count_after = {}
#    dic_count_variables_aberrantes = {}
#
#    print('Nombre de valeurs aberrantes :\n')
#    for variable in columns:
#        dic_count_before[variable] = dataframe[variable].value_counts().sum()
#        dataframe[variable] = [t if t<value else np.NaN for t in dataframe[variable]]
#        dic_count_after[variable] = dataframe[variable].value_counts().sum()
#        dic_count_variables_aberrantes[variable] = dic_count_before[variable] - #dic_count_after[variable]
#        
#    return dic_count_variables_aberrantes


#def below_zero_filter(x):
#    if x < 0:
#        x = np.NaN
#    else:
#        None

#Fonction pour le filtre de séries ou colonnes d'un dataframe
#def below_value_filter(x,value):
#    """Filtre les valeurs en dessous d'une valeur définie. A associer avec .apply(lambda x: ) pour modifier les colonnes 
#    d'un dataframe"""
#    import numpy as np
#    if x != np.NaN and x < value:
#        return np.NaN
#    else:
#        return x
        
#Fonction pour le filtre de séries ou colonnes d'un dataframe
#def above_value_filter(x,value):
#    """Filtre les valeurs au dessus d'une valeur définie. A associer avec .apply(lambda x: ) pour modifier les colonnes 
#    d'un dataframe"""
#    import numpy as np
#    if x != np.NaN and x > value:
#        return np.NaN
#    else:
#        return x

# Calcul du nombre de valeurs aberrantes d'un dataframe, filtre de celle-ci et impression du résultat
#Input: une SEULE valeur au choix pour chaque colonne
def high_outlier_filter_df(dataframe, columns, value):
    """Calcul du nombre de valeurs aberrantes d'un dataframe au-dessus d'une valeur, filtre de celles-ci et impression 
    du résultat"""
    import numpy as np
    dataframe_filter = dataframe.copy()
    dic_count_before = {}
    dic_count_after = {}
    dic_count_variables_aberrantes = {}
    dic_percent_variables_aberrantes = {}

    print('Nombre de valeurs aberrantes :\n')
    for variable in columns:
        dic_count_before[variable] = dataframe_filter[variable].value_counts().sum()
        dataframe_filter[variable] = dataframe_filter[variable].map(lambda x: x if x<value else np.NaN)
        dic_count_after[variable] = dataframe_filter[variable].value_counts().sum()
        dic_count_variables_aberrantes[variable] = dic_count_before[variable] - dic_count_after[variable]
        dic_percent_variables_aberrantes[variable] = (dic_count_before[variable] - 
                                                      dic_count_after[variable])/len(dataframe)
        
    print(dic_count_variables_aberrantes)  
    print('\n')
    print('ratio de valeurs aberrantes :\n')
    print(dic_percent_variables_aberrantes)    
    return dataframe_filter

# Calcul du nombre de valeurs aberrantes d'un dataframe, filtre de celle-ci et impression du résultat
#Input: une SEULE valeur au choix pour chaque colonne
def low_outlier_filter_df(dataframe, columns,value):
    """Calcul du nombre de valeurs aberrantes d'un dataframe au-dessus d'une valeur, filtre de celles-ci et impression du 
    résultat"""
    import numpy as np
    dic_count_before = {}
    dic_count_after = {}
    dic_count_variables_aberrantes = {}
    dic_percent_variables_aberrantes = {}

    print('Nombre de valeurs aberrantes :\n')
    for variable in columns:
        dic_count_before[variable] = dataframe[variable].value_counts().sum()
        dataframe[variable] = dataframe[variable].map(lambda x: x if x>value else np.NaN)
        dic_count_after[variable] = dataframe[variable].value_counts().sum()
        dic_count_variables_aberrantes[variable] = dic_count_before[variable] - dic_count_after[variable]
        dic_percent_variables_aberrantes[variable] = (dic_count_before[variable] - 
                                                      dic_count_after[variable])/len(dataframe)
        
    print(dic_count_variables_aberrantes)  
    print('\n')
    print('ratio de valeurs aberrantes :\n')
    print(dic_percent_variables_aberrantes) 
    return dataframe

def sign_invert_filter_df(dataframe, columns):
    """Inversion de signe de valeurs numériques si inférieur à zéro"""
    import numpy as np
    dic_count_before = {}
    dic_count_after = {}
    dic_count_variables_aberrantes = {}
    dic_percent_variables_aberrantes = {}

    print('Nombre de valeurs aberrantes :\n')
    for variable in columns:
        dic_count_before[variable] = dataframe[variable].value_counts().sum()
        dataframe[variable] = dataframe[variable].map(lambda x: -x if x<0 else x)
        dic_count_after[variable] = dataframe[variable].value_counts().sum()
        dic_count_variables_aberrantes[variable] = dic_count_before[variable] - dic_count_after[variable]
        dic_percent_variables_aberrantes[variable] = (dic_count_before[variable] - 
                                                      dic_count_after[variable])/len(dataframe)
        
    print(dic_count_variables_aberrantes)  
    print('\n')
    print('ratio de valeurs aberrantes :\n')
    print(dic_percent_variables_aberrantes) 
    return dataframe


# Calcul du nombre de valeurs aberrantes d'un dataframe, filtre de celle-ci et impression du résultat
# Input: dictionnaire (une valeur par colonne)
def dic_high_outlier_filter_df(dataframe, columns, dictionary):
    """Calcul du nombre de valeurs aberrantes d'un dataframe au-dessus d'une valeur lue dans un dictionnaire, filtre de 
    celles-ci et impression du résultat"""
    import numpy as np
    dic_count_before = {}
    dic_count_after = {}
    dic_count_variables_aberrantes = {}
    dic_percent_variables_aberrantes = {}
    
    print('Nombre de valeurs aberrantes :\n')
    for variable in columns:
        dic_count_before[variable] = dataframe[variable].value_counts().sum()
        dataframe[variable] = dataframe[variable].map(lambda x: x if x<dictionary[variable] else np.NaN)
        dic_count_after[variable] = dataframe[variable].value_counts().sum()
        dic_count_variables_aberrantes[variable] = dic_count_before[variable] - dic_count_after[variable]
        dic_percent_variables_aberrantes[variable] = (dic_count_before[variable] - 
                                                      dic_count_after[variable])/len(dataframe)
        
    print(dic_count_variables_aberrantes)  
    print('\n')
    print('ratio de valeurs aberrantes :\n')
    print(dic_percent_variables_aberrantes)    
    return dataframe

#-------------------------------------------------------------------------------------------------------------------------
# Recherche et filtre de chaînes de caractères dans dataframe
#-------------------------------------------------------------------------------------------------------------------------

# Renvoie un dataframe filtré
def word_column_filter_df(dataframe, column_to_filter, column_freeze, word_list):
# La fonction .where() donne une position qu'il faut transformer en index
# Il faut entrer le nom d'une colonne repère (exemple: code produit) pour retrouver l'index, ou construire un colonne de re-indexée.
    """Filtre les colonnes d'un dataframe, en fonction d'une liste de mots, puis retourne le dataframe"""
    import re
    position_to_drop_lst = np.where(dataframe[column_to_filter].str.contains('|'.join(map(re.escape, word_list)), 
                                                                           np.NaN))[0]
    indices_to_drop_lst = []
    for position in position_to_drop_lst:
        indice = (dataframe[dataframe[column_freeze] == dataframe.iloc[position].loc[column_freeze]]).index[0]
        indices_to_drop_lst.append(indice)

    print("Nombre de lignes supprimées:")
    nbr= len(indices_to_drop_lst)
    print(nbr)
    print("\n")

    dataframe.drop(indices_to_drop_lst, axis=0,inplace=True)

    return dataframe

# Renvoie une liste des indices
def word_column_filter_lst(dataframe, column_to_filter, column_freeze, word_list):
# La fonction .where() donne une position qu'il faut transformer en index
# Il faut entrer le nom d'une colonne repère (exemple: code produit) pour retrouver l'index, ou construire un colonne de re-indexée.
    """Filtre les colonnes d'un dataframe, en fonction d'une liste de mots, puis retourne le dataframe"""
    import re
    position_to_drop_lst = np.where(dataframe[column_to_filter].str.contains('|'.join(map(re.escape, word_list)), 
                                                                           np.NaN))[0]
    indices_to_drop_lst = []
    for position in position_to_drop_lst:
        indice = (dataframe[dataframe[column_freeze] == dataframe.iloc[position].loc[column_freeze]]).index[0]
        indices_to_drop_lst.append(indice)

    print("Nombre de lignes supprimées:")
    nbr= len(indices_to_drop_lst)
    print(nbr)
    print("\n")

    return indices_to_drop_lst

#-------------------------------------------------------------------------------------------------------------------------
# Tirage aléatoire de produits/bâtiments etc 
#-------------------------------------------------------------------------------------------------------------------------
def random_item(df, item_column):
    """Tirage aléatoire de lignes de dataframe et renvoie un nouveau dataframe"""
   
    new_df = pd.DataFrame([], columns=df.columns)
    building_lst = df[item_column].unique()
    indices_keep = []

    for building in building_lst:

        # Filtre du dataframe sur les lignes ayant ce nom de bâtiment
        building_df = df[df[item_column] == building]
        # Sélection unique
        #building_count.append(building)

        # Tirage aléatoite d'un indice de ligne sur ce dataframe filtré
        year_building_ind = list(np.random.choice(list(building_df.index), 1))[0]
        indices_keep.append(year_building_ind)

        # Création d'un dataframe de ligne à ajouter
        #raw_to_add = pd.DataFrame(df.iloc[year_building_ind]).T

        # Concaténation avec le nouveau dataframe
        #new_df = pd.concat([new_df, raw_to_add],axis=0)
        
        # Suppression des indices de lignes 
        new_df = df.iloc[indices_keep, range(len(df.columns))]

    return new_df

#-------------------------------------------------------------------------------------------------------------------------
# Comparaison de listes
#-------------------------------------------------------------------------------------------------------------------------
def common_elements(list1, list2):
    """Check of common values of two lists"""
    
    print(f"Liste 1: {len(list1)}")
    print(f"Liste 2: {len(list2)}")
    
    print("\n")
    print(f"Common elements: ")
    
    return list(set(list1) & set(list2))

def separate_elements(list1, list2):
    """Check of different values of two lists"""
    
    print(f"Liste 1: {len(list1)}")
    print(f"Liste 2: {len(list2)}")
    
    print("\n")
    print(f"Separate elements: ")
    
    return list(set(list1) ^ set(list2))