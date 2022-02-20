# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:31:59 2022

@author: fabio
"""


from datetime import datetime
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
import plotly.express as px
import pathlib
import os

#import warnings
#warnings.filterwarnings("ignore")

#%% start up run


script_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\"
database_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\data\\"
metadata_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\Metadata\\"

dataprep_filename = "programming_analysis_project_data_prep.py"
analysis_filename = "programming_analysis_project_analysis_file.py"


#import prepared data and methods from [programming_analysis_project_data_prep.py]
runcell("start up run", script_filepath + dataprep_filename)
runcell("General Methods", script_filepath + dataprep_filename)
#runcell("Import Additional Modules and Define General Parameters", script_filepath + analysis_filename)




#%% Create Analysis Printout

runcell("dud actors study", script_filepath + metadata_filepath)


#%% Create Model


runcell("Create Forecase Model Input", script_filepath + metadata_filepath)


#%% Create Forecase Model Input

runcell('General Methods', 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/IMDB-Stars-Project/programming_analysis_project_analysis_file.py')

failures_to_populate_films_in_model_input = 0
failure_to_populate_actor_nconst_for_metascore_generation = 0
progress = 0
film_years_approximated_as_2020 = 0

#Convert the relational table of actor to film to a format required for the following operation
md_title_principals_reduced_input = md_title_principals_reduced.copy()
md_title_principals_reduced_input.set_index(['nconst', 'tconst'], inplace=True)
md_title_principals_reduced_input = md_title_principals_reduced_input.loc[:, ~md_title_principals_reduced_input.columns.str.contains('^Unnamed')]


print(str(datetime.now()))
print("1")

failures_to_populate_films_in_model_input = 0
failure_to_populate_actor_nconst_for_metascore_generation = 0
progress = 0
film_years_approximated_as_2020 = 0                                                                    
training_tconsts = loadtxt(metadata_filepath + 'training_tconsts.csv', delimiter=',', dtype=object)
testing_tconsts = loadtxt(metadata_filepath + 'testing_tconsts.csv', delimiter=',', dtype=object)

print(str(datetime.now()))
print("gen X_train_shor")
X_train_short, Y_train_short = populate_actor_metascores_for_insertion_into_the_model(training_tconsts[0::10], 6, md_PrimaryActorsList, md_secondary_actors, md_title_principals_reduced)
savetxt(metadata_filepath + 'X_train_short.csv', asarray(X_train_short), delimiter=',', fmt='%s')
savetxt(metadata_filepath + 'Y_train_short.csv', asarray(Y_train_short), delimiter=',', fmt='%s')
print(str(datetime.now()))
X_test_short, Y_test_short = populate_actor_metascores_for_insertion_into_the_model(testing_tconsts[0::10], 6, md_PrimaryActorsList, md_secondary_actors, md_title_principals_reduced)
savetxt(metadata_filepath + 'X_test_short.csv', asarray(X_train_short), delimiter=',', fmt='%s')
savetxt(metadata_filepath + 'Y_test_short.csv', asarray(Y_train_short), delimiter=',', fmt='%s')


print(str(datetime.now()))
print("gen X_train")
X_train, Y_train = populate_actor_metascores_for_insertion_into_the_model(training_tconsts, 6, md_PrimaryActorsList, md_secondary_actors, md_title_principals_reduced)
savetxt(metadata_filepath + 'X_train.csv', asarray(X_train), delimiter=',', fmt='%s')
savetxt(metadata_filepath + 'Y_train.csv', asarray(Y_train), delimiter=',', fmt='%s')
X_train, Y_train = populate_actor_metascores_for_insertion_into_the_model(testing_tconsts, 6, md_PrimaryActorsList, md_secondary_actors, md_title_principals_reduced)
savetxt(metadata_filepath + 'X_test.csv', asarray(X_train), delimiter=',', fmt='%s')
savetxt(metadata_filepath + 'Y_test.csv', asarray(Y_train), delimiter=',', fmt='%s')

print(str(datetime.now()))
print("3")

from sklearn.linear_model import LinearRegression
regressor_short_x = LinearRegression()
regressor_x = LinearRegression()

print(str(datetime.now()))
print("4")

regressor_short_x.fit(X_train_short, Y_train_short)
regressor_x.fit(X_train, Y_train)

print(str(datetime.now()))
print("5")
# Predicting the Test set results
Y_pred_short = regressor_short_x.predict(X_test_short)
Y_pred       = regressor_x.predict(X_test)

print(str(datetime.now()))
print("6")
from sklearn.metrics import r2_score
score_short = r2_score(Y_test_short,Y_pred_short)
score       = r2_score(Y_test,Y_pred)

print(str(datetime.now()))






#%% Table info and clean-up

md_PrimaryActorsList = md_PrimaryActorsList.loc[:, ~md_PrimaryActorsList.columns.str.contains('^Unnamed*')]
md_actor_to_film.info()
md_film_scores.info()
md_PrimaryActorsList.info()
md_secondary_actors.info()
md_title_principals_reduced.info()


#%% md_PrimaryActorsList - Basic Analysis

md_PrimaryActorsList_2 = md_PrimaryActorsList.drop(["Number", "name", "nconst", "Ratings", "Rating Years", "tconst", "Films_Training_Qty", "Films_Testing_Qty"], axis=1)
md_PrimaryActorsList_corr = md_PrimaryActorsList_2.corr()
sns.pairplot(md_PrimaryActorsList_2)


#%% md_secondary_actors - Basic Analysis

md_secondary_actors_2 = md_secondary_actors.drop(["name", "nconst", "tconst"], axis=1)
md_secondary_actors_corr = md_secondary_actors_2.corr()
md_secondary_actors_pairplot = sns.pairplot(md_secondary_actors_2, plot_kws=dict(marker=".", linewidth=1))
md_secondary_actors_pairplot.fig.suptitle("md_secondary_actors")
#md_secondary_actors_pairplot.fig.show()

#%%


sns.pairplot(md_secondary_actors_2, kind="hist", diag_kws = {'alpha':1, 'bins':10})




"""
sns.set_theme(style="white")
g = sns.PairGrid(md_secondary_actors_2, diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
"""




#%% md_film_scores - Basic Analysis

md_film_scores_2 = md_film_scores.drop(["name", "genre", "nconst", "Primary Actor Relative Ratings"], axis=1)
md_film_scores_corr = md_film_scores_2.corr()
md_film_scores_pairplot = sns.pairplot(md_film_scores_2)
md_film_scores_pairplot.fig.suptitle("md_film_scores")
#md_secondary_actors_pairplot.fig.show()







#%% dud actors study


md_secondary_actors_desc = md_secondary_actors.describe()
md_secondary_actors_names_dropped = md_secondary_actors.drop(["name", "nconst"], axis=1)

md_secondary_actors_least_20_films_mask = md_secondary_actors_names_dropped["Count"] >= 20
md_secondary_actors_least_20_films  = md_secondary_actors_names_dropped[md_secondary_actors_least_20_films_mask]
md_film_scores_pairplot = sns.pairplot(md_secondary_actors_least_20_films)
md_film_scores_pairplot.fig.suptitle("md_film_scores_least_20_films")

temp_mask = (md_secondary_actors_names_dropped["Count"] >= 20) & (md_secondary_actors_names_dropped["Relative Actor Score - Mean"] >=  0.5)
md_secondary_actors_least_20_films_relative_rating_below_minus05 = md_secondary_actors_names_dropped[temp_mask]
md_secondary_actors_least_20_films_relative_rating_below_minus05["Cat"] = "Below -0.5"
temp_mask = (md_secondary_actors_names_dropped["Count"] >= 20) & (md_secondary_actors_names_dropped["Relative Actor Score - Mean"] <=  -0.5)
md_secondary_actors_least_20_films_relative_rating_above_05 = md_secondary_actors_names_dropped[temp_mask]
md_secondary_actors_least_20_films_relative_rating_above_05["Cat"] = "Above +0.5"

#temp_mask = (md_secondary_actors_names_dropped["Count"] >= 20) & ((md_secondary_actors_names_dropped["Relative Actor Score - Mean"] <=  -0.5) | (md_secondary_actors_names_dropped["Relative Actor Score - Mean"] >=  0.5))
md_secondary_actors_least_20_films_relative_rating_above_below_05 = md_secondary_actors_least_20_films_relative_rating_below_minus05.append(md_secondary_actors_least_20_films_relative_rating_above_05)

print(len(md_secondary_actors_least_20_films_relative_rating_below_minus05))
print(len(md_secondary_actors_least_20_films_relative_rating_above_05))
print(len(md_secondary_actors_least_20_films_relative_rating_above_below_05))


a = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_below_minus05, hue="Relative Actor Score - Mean", palette="vlag")
a.fig.suptitle("md_secondary_actors_least_20_films_relative_rating_below_minus05")

b = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_above_05, hue="Relative Actor Score - Mean", palette="vlag")
b.fig.suptitle("md_secondary_actors_least_20_films_relative_rating_above_05")

ab = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_above_below_05, hue="Cat", palette="vlag")
#ab.fig.suptitle("md_secondary_actors_least_20_films_relative_rating_above_below_05")

abc = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_above_below_05[['Relative Actor Score - Mean', 'Relative Actor Score - Std', "Cat"]], hue="Cat", palette="vlag")


p=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)



#%% Draft - Parallel Axis Chart

fig2 = px.parallel_coordinates(md_PrimaryActorsList_2)
fig.show()


#%% General Methods

def populate_actor_metascores_for_insertion_into_the_model(film_tconst_list, desired_number_of_actors, primary_actor_DB, secondary_actor_DB,  md_title_principals_relational):
    """ these variables have been removed md_actor_to_film, md_actor_to_film_secondary):"""
    
    
    #this method is the wrapper that repeatability creates the inputs for the regression models.
    #It also outputs a separate dfs documenting the film ratings
    
    global failures_to_populate_films_in_model_input, progress, film_years_approximated_as_2020
    counter = 0
    start = datetime.now()
    input_column_values = np.array([])
    rating_string = "rating"
    start = datetime.now()
    #md_film_scores_with_tconst_index =  md_film_scores.set_index([['tconst'], inplace=False])
    
    for i in range(0, desired_number_of_actors):
        input_column_values = np.append(input_column_values, "metascore_" + str(i))
    
    input_column_values = np.append(input_column_values, rating_string)
    output = pd.DataFrame(columns = input_column_values)
       
    for film_tconst in film_tconst_list:
        
        counter += 1
        fg_counter(counter, len(film_tconst_list), 100, start, False)
        
        
        
        #nconsts = populate_input_actor_scores_for_film           (film_tconst, desired_number_of_actors, primary_actor_DB, secondary_actor_DB, md_title_principals_relational)
        nconsts = return_nconsts_in_film_with_highest_film_counts(film_tconst, desired_number_of_actors, primary_actor_DB, secondary_actor_DB, md_title_principals_relational)
        '''work around add to return a film year of 2020 if a film year couldnt be found'''
        if len(nconsts) > 0:
            try:
                temp_film_year = md_film_scores['film year'][film_tconst]
            except KeyError as err:
                temp_film_year = 2020
                film_years_approximated_as_2020 += 1
            meta_scores = generate_entry_for_score_predictor_from_nconst(nconsts, desired_number_of_actors, temp_film_year , primary_actor_DB, secondary_actor_DB, md_title_principals_relational)        
            try:
                data = np.append(meta_scores, md_film_scores.loc[md_film_scores['tconst'] == film_tconst]['rating'].values[0])
                addition = pd.Series(data=data, index = input_column_values)
                output = output.append(addition, ignore_index=True)
            except IndexError as err:
                failures_to_populate_films_in_model_input += 1
    
    return output.drop(rating_string, axis = 1), output[rating_string]

                  
def return_nconsts_in_film_with_highest_film_counts(film_tconst, actors_qty, primary_actor_DB, secondary_actor_DB, md_title_principals_reduced_input):
    # called twice by populate_input_actor_scores_for_film
    
    global failure_to_populate_actor_nconst_for_metascore_generation
    
    nconsts = np.array([])

    
    #primary_actors_in_film = md_actor_to_film_relational_DB[md_actor_to_film_relational_DB.index.get_level_values('tconst') == film_tconst]
    
    
    #primary_actors_in_film = pd.DataFrame(columns  = md_title_principals_reduced_input.columns)
    
    primary_actors_in_film = md_title_principals_reduced_input[pd.DataFrame(md_title_principals_reduced_input.tconst.tolist()).isin([film_tconst]).any(1).values]
    
    #primary_actors_in_film = md_title_principals_reduced_input[md_title_principals_reduced_input.index.get_level_values('tconst') == film_tconst]
    primary_actors_in_film = primary_actors_in_film.drop_duplicates()

    if not 'count' in primary_actors_in_film.columns:
        primary_actors_in_film['count'] = None
    
    #assign a count value to a actor if one not populated yet
    for index in primary_actors_in_film.index:
        if primary_actors_in_film['count'][index] == None:
            # if not found in primary actor DB, try secondary
            actor_nconst = primary_actors_in_film['nconst'][index]
            actor_record = return_actor_record_from_nconst_v2(actor_nconst)
            actor_name = actor_record[0]['name'].values[0]
            #if len(primary_actor_DB.loc[primary_actor_DB['name'] == primary_actors_in_film['actor name'][index]]) > 0:
            if len(primary_actor_DB.loc[primary_actor_DB['name'] == actor_name]) > 0:
                #primary_actors_in_film.loc[:,('count', index)] = len(primary_actor_DB.loc[primary_actor_DB['name'] == actor_name])
                
                tconst_list_outer_list = primary_actor_DB.loc[primary_actor_DB['name'] == actor_name]['tconst']
                primary_actors_in_film.loc[:,'count'][index]  = 0
                for list_O in tconst_list_outer_list:
                    primary_actors_in_film.loc[:,'count'][index] += len(list_O)
                
                
            #elif len(secondary_actor_DB.loc[secondary_actor_DB['name'] == primary_actors_in_film['actor name'][index]]) > 0:
            elif len(secondary_actor_DB.loc[secondary_actor_DB['name'] == actor_name]) > 0:
                tconst_list_outer_list = secondary_actor_DB.loc[secondary_actor_DB['name'] == actor_name]['tconst']
                """tconst_list  = np.array([])
                for i in tconst_list_outer_list:
                    tconst_list = np.append(tconst_list, value)"""
                primary_actors_in_film.loc[:,'count'][index]  = 0
                for list_O in tconst_list_outer_list:
                    primary_actors_in_film.loc[:,'count'][index] += len(list_O)
            else:
                #actor record not found. Drop for this film and count drop
                primary_actors_in_film = primary_actors_in_film.drop(index )
                failure_to_populate_actor_nconst_for_metascore_generation += 1
                    
    actors_to_be_populated_qty = min(actors_qty, len(primary_actors_in_film))
    #actors_to_populate_nconsts = np.array([])
    if film_tconst == ('tt0032285'):
        print("Step")
    
    
    for i in range(0, actors_to_be_populated_qty):
        max_count = primary_actors_in_film['count'].max()
        target_index = primary_actors_in_film.loc[primary_actors_in_film['count'] == max_count].index
        if len(target_index) > 1:
            target_index = target_index[0]
        nconsts = np.append(nconsts, primary_actors_in_film['nconst'][target_index])
        primary_actors_in_film = primary_actors_in_film.drop(target_index)
    
    return nconsts
    
            
    
    
    
    """
    #'''Work around to stop populating actor nconsts if more then one nconsts was deleted previously due to the multi-index nature of the method, also because the '''
    for i in range(0, actors_to_be_populated_qty):
        
        try:
            #a = target_index[0][0]
            max_count = primary_actors_in_film['count'].max()
            target_index = primary_actors_in_film.loc[primary_actors_in_film['count'] == max_count].index
            nconsts = np.append(nconsts, target_index[0][0])
            primary_actors_in_film = primary_actors_in_film.drop(target_index)
        except IndexError as err:
            failure_to_populate_actor_nconst_for_metascore_generation += 1
    
    return nconsts"""



def String_Convert(string):
    li = list(string.split(" "))
    return li

def generate_entry_for_score_predictor_from_nconst(nconsts, column_qty, year, primary_actor_list, secondary_actor_list, md_title_principals_relational):
    meta_scores = np.array([])
    
    global failures_to_populate_films_in_model_input
    
    for nconst in nconsts:
        #check if nconst is a primary actor
        nconst_is_primary = False
        nconst_is_secondary = False
        #determine if nconst, belongs to a primary actor and return the row number
        for primary_actor_row in range(0, len(primary_actor_list)):
            if nconst in primary_actor_list['nconst'][primary_actor_row]:
                row_num = primary_actor_row
                nconst_is_primary = True
                break
        
        #determine if nconst, belongs to a secondary actor and return the row number
        if nconst_is_primary == False:
            _secondary_actor_row = secondary_actor_list.loc[secondary_actor_list['nconst'] == nconst]
            #secondary_actor_row = _secondary_actor_row.loc[_secondary_actor_row.index[0]]
            secondary_actor_row = _secondary_actor_row.loc[_secondary_actor_row.index]
            nconst_is_secondary = True
            
            
            
        if nconst_is_primary == True:
            #if the nconst refers to a primary actor, their metascore for a film is generated according to their linear model and the films year
            expected_rating = (2020 - year) * md_PrimaryActorsList["Model_Gradient"][row_num] + md_PrimaryActorsList["Model Rating 2020"][row_num]
            meta_scores = np.append(meta_scores, expected_rating)
        elif nconst_is_secondary == True:
            #otherwise thier metascore is just their average score of the dataset
            ratings = np.array([])
            md_film_scores_temp = md_film_scores.set_index('tconst',inplace=False)
            try:
                for tconst in relister_single_functional(secondary_actor_row['tconst'], "string"):
                    try:
                        ratings = np.append(ratings, md_film_scores_temp['rating'][tconst])
                    except KeyError as err:
                        failures_to_populate_films_in_model_input += 1
            except TypeError as err:
                failures_to_populate_films_in_model_input += 1
            expected_rating = ratings.mean()
            meta_scores = np.append(meta_scores, expected_rating)
        
        
            
    
    zeros_to_add_qty = column_qty - len(meta_scores)
    for i in range(0, zeros_to_add_qty):
        meta_scores = np.append(meta_scores, [0])
    
    return meta_scores

                                                            

        
    
