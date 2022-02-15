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



#%% start up run


script_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\"
database_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\data\\"
metadata_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\Metadata\\"

dataprep_filename = "programming_analysis_project_data_prep.py"
analysis_filename = "programming_analysis_project_analysis_file.py"


#import prepared data and methods from [programming_analysis_project_data_prep.py]
runcell("start up run", script_filepath + dataprep_filename)
runcell("Import Additional Modules and Define General Parameters", script_filepath + analysis_filename)


#%% Create Analysis Printout

runcell("dud actors study", script_filepath + metadata_filepath)


#%% Create Model


runcell("Create Forecase Model Input", script_filepath + metadata_filepath)


#%% Create Forecase Model Input

film_tconst_list = ['tt1067106', 'tt2008009'] #film is a christmas carol
X, Y = populate_actor_metascores_for_insertion_into_the_model(film_tconst_list, 3, 3, md_PrimaryActorsList, md_secondary_actors, md_actor_to_film, md_actor_to_film_secondary)




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
md_secondary_actors_pairplot = sns.pairplot(md_secondary_actors_2)
md_secondary_actors_pairplot.fig.suptitle("md_secondary_actors")
#md_secondary_actors_pairplot.fig.show()


#%% md_film_scores - Basic Analysis

md_film_scores_2 = md_film_scores.drop(["name", "genre", "nconst", "Primary Actor Relative Ratings"], axis=1)
md_film_scores_corr = md_film_scores_2.corr()
md_film_scores_pairplot = sns.pairplot(md_film_scores_2)
md_film_scores_pairplot.fig.suptitle("md_film_scores")
#md_secondary_actors_pairplot.fig.show()


#%%




#%% dud actors study


md_secondary_actors_desc = md_secondary_actors.describe()
md_secondary_actors_least_20_films_mask = md_secondary_actors["Count"] >= 20
md_secondary_actors_least_20_films  = md_secondary_actors[md_secondary_actors_least_20_films_mask]
md_film_scores_pairplot = sns.pairplot(md_secondary_actors_least_20_films)
md_film_scores_pairplot.fig.suptitle("md_film_scores_least_20_films")

temp_mask = (md_secondary_actors["Count"] >= 20) & (md_secondary_actors["Relative Actor Score - Mean"] >=  0.5)
md_secondary_actors_least_20_films_relative_rating_below_minus05 = md_secondary_actors[temp_mask]
md_secondary_actors_least_20_films_relative_rating_below_minus05["Cat"] = 1
temp_mask = (md_secondary_actors["Count"] >= 20) & (md_secondary_actors["Relative Actor Score - Mean"] <=  -0.5)
md_secondary_actors_least_20_films_relative_rating_above_05 = md_secondary_actors[temp_mask]
md_secondary_actors_least_20_films_relative_rating_above_05["Cat"] = 0

#temp_mask = (md_secondary_actors["Count"] >= 20) & ((md_secondary_actors["Relative Actor Score - Mean"] <=  -0.5) | (md_secondary_actors["Relative Actor Score - Mean"] >=  0.5))
md_secondary_actors_least_20_films_relative_rating_above_below_05 = md_secondary_actors_least_20_films_relative_rating_below_minus05.append(md_secondary_actors_least_20_films_relative_rating_above_05)

print(len(md_secondary_actors_least_20_films_relative_rating_below_minus05))
print(len(md_secondary_actors_least_20_films_relative_rating_above_05))
print(len(md_secondary_actors_least_20_films_relative_rating_above_below_05))


a = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_below_minus05, hue="Relative Actor Score - Mean", palette="vlag")
a.fig.suptitle("md_secondary_actors_least_20_films_relative_rating_below_minus05")

b = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_above_05, hue="Relative Actor Score - Mean", palette="vlag")
b.fig.suptitle("md_secondary_actors_least_20_films_relative_rating_above_05")

ab = sns.pairplot(md_secondary_actors_least_20_films_relative_rating_above_below_05, hue="Cat", palette="vlag")
ab.fig.suptitle("md_secondary_actors_least_20_films_relative_rating_above_below_05")



p=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)






#%% Draft - Parallel Axis Chart

fig2 = px.parallel_coordinates(md_PrimaryActorsList_2)
fig.show()


#%% General Methods

def populate_input_actor_scores_for_film(film_tconst, desired_number_of_primary_actors, desired_number_of_secondary_actors, primary_actor_DB, secondary_actor_DB, md_actor_to_film, md_actor_to_film_secondary):
    #Returns a list of primary and secondary actors nconst related to a film, to later populate a row of the predictor
    """ find a way to filter according to index"""
    
    '''Return up to [desired_number_of_primary_actors]  qty of tconsts'''
    nconsts = return_nconsts_in_film_with_highest_film_counts(film_tconst, desired_number_of_primary_actors, primary_actor_DB, [], md_actor_to_film)
    #print("a")
    '''Return up to X  qty of tconsts to ensure that the total number of tconsts equals the sum of decired primary and secondary actors'''
    secondary_actors_to_populate = desired_number_of_primary_actors + desired_number_of_secondary_actors - len(nconsts)
    nconsts = return_nconsts_in_film_with_highest_film_counts(film_tconst, secondary_actors_to_populate, md_secondary_actors, nconsts, md_actor_to_film_secondary)
    
    return nconsts
        
def return_nconsts_in_film_with_highest_film_counts(film_tconst, actors_qty, actor_DB, previous_nconst, md_actor_to_film_relational_DB):
    # called twice by populate_input_actor_scores_for_film
    if len(previous_nconst) == 0:
        nconsts = np.array([])
    else:
        nconsts = previous_nconst
    
    
    primary_actors_in_film = md_actor_to_film_relational_DB[md_actor_to_film_relational_DB.index.get_level_values('tconst') == film_tconst]
    if not 'count' in primary_actors_in_film.columns:
        primary_actors_in_film['count'] = None
    
    for index in primary_actors_in_film.index:
        if primary_actors_in_film['count'][index] == None:
            row_in_df = actor_DB.loc[actor_DB['name'] == primary_actors_in_film['actor name'][index]].index[0]
            #primary_actors_in_film['count'][index] = len(md_PrimaryActorsList['tconst'][row_in_df])
            primary_actors_in_film.loc[index, 'count'] = len(actor_DB['tconst'][row_in_df])
        
    #remove tconsts already counted
    for row_num in range(len(primary_actors_in_film) - 1, -1, -1):
        if primary_actors_in_film.index.get_level_values('nconst')[row_num] in nconsts:
            primary_actors_in_film = primary_actors_in_film.drop(primary_actors_in_film.index[row_num])
        try:
            for i in range(0, len(primary_actors_in_film.index.get_level_values('nconst')[row_num])):
                if primary_actors_in_film.index.get_level_values('nconst')[row_num][i] in nconsts:
                    primary_actors_in_film = primary_actors_in_film.drop(primary_actors_in_film.index[row_num])
                    i = len(primary_actors_in_film.index.get_level_values('nconst')[row_num])
        except KeyError as err:
            name = "w"
    
    actors_to_be_populated_qty = min(actors_qty, len(primary_actors_in_film))
    #actors_to_populate_nconsts = np.array([])
    
    for i in range(0, actors_to_be_populated_qty):
        max_count = primary_actors_in_film['count'].max()
        target_index = primary_actors_in_film.loc[primary_actors_in_film['count'] == max_count].index
        nconsts = np.append(nconsts, target_index[0][0])
        primary_actors_in_film = primary_actors_in_film.drop(target_index)
    
    return nconsts

def generate_entry_for_score_predictor_from_nconst(nconsts, column_qty, year, primary_actor_list, secondary_actor_list, md_actor_to_film, md_actor_to_film_secondary):
    meta_scores = np.array([])
    for i in range(0, len(nconsts)):
        #check if nconst is a primary actor
        nconst_is_primary = False
        #determine if nconst, belongs to a primary actor and return the row number
        for primary_actor_row in range(0, len(primary_actor_list)):
            if nconsts[i] in primary_actor_list['nconst'][primary_actor_row]:
                row_num = primary_actor_row
                nconst_is_primary = True
                break
        
        #determine if nconst, belongs to a secondary actor and return the row number
        if nconst_is_primary == False:
            for secondary_actor_row in range(0, len(secondary_actor_list)):
                if nconsts[i] in secondary_actor_list['nconst'] == True:
                    row_num = secondary_actor_row
                    break
            
        if nconst_is_primary == True:
            #if the nconst refers to a primary actor, their metascore for a film is generated according to their linear model and the films year
            expected_rating = (2020 - year) * md_PrimaryActorsList["Model_Gradient"][row_num] + md_PrimaryActorsList["Model Rating 2020"][row_num]
        else:
            #otherwise thier metascore is just their average score of the dataset
            ratings = np.array([])
            for tconst in secondary_actor_list['tconst']:
                ratings = np.append(ratings, md_film_scores['rating'][tconst])
            expected_rating = ratings.mean()
        
        meta_scores = np.append(meta_scores, expected_rating)
        
    for i in range(0, column_qty - len(nconsts)):
        meta_scores = np.append(meta_scores, [0])
    
    return meta_scores

    
def populate_actor_metascores_for_insertion_into_the_model(film_tconst_list, desired_number_of_primary_actors, desired_number_of_secondary_actors, primary_actor_DB, secondary_actor_DB, md_actor_to_film, md_actor_to_film_secondary):
    
    #this method is the wrapper that repeatability creates the inputs for the regression models.
    #It also outputs a separate dfs documenting the film ratings
    total_actor_metascores_qty = desired_number_of_primary_actors + desired_number_of_secondary_actors
    input_column_values = np.array([])
    rating_string = "rating"
    
    for i in range(0, total_actor_metascores_qty):
        input_column_values = np.append(input_column_values, "metascore_" + str(i))
    
    input_column_values = np.append(input_column_values, rating_string)
    output = pd.DataFrame(columns = input_column_values)
       
    for film_tconst in film_tconst_list:
        nconsts = populate_input_actor_scores_for_film(film_tconst, desired_number_of_primary_actors, desired_number_of_secondary_actors, primary_actor_DB, secondary_actor_DB, md_actor_to_film, md_actor_to_film_secondary)
        meta_scores = generate_entry_for_score_predictor_from_nconst(nconsts, total_actor_metascores_qty, md_film_scores['film year'][film_tconst], primary_actor_DB, secondary_actor_DB, md_actor_to_film, md_actor_to_film_secondary)        
        data = np.append(meta_scores, md_film_scores['rating'][film_tconst])
        addition = pd.Series(data=data, index = input_column_values)
        output = output.append(addition, ignore_index=True)
    
    
    
    return output.drop(rating_string, axis = 1), output[rating_string]
        
    
