# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:31:59 2022

@author: fabio
"""






#%% start up run


script_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\"
database_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\data\\"
metadata_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\Metadata\\"

dataprep_filename = "programming_analysis_project_data_prep.py"
analysis_filename = "programming_analysis_project_analysis_file.py"


#import prepared data and methods from [programming_analysis_project_data_prep.py]
runcell("start up run", script_filepath + dataprep_filename)
runcell("Import Additional Modules and Define General Parameters", script_filepath + analysis_filename)


#%% Import Additional Modules and Define General Parameters



#%% Table info and clean-up

md_PrimaryActorsList = md_PrimaryActorsList.loc[:, ~md_PrimaryActorsList.columns.str.contains('^Unnamed*')]
md_actor_to_film.info()
md_film_scores.info()
md_PrimaryActorsList.info()
md_secondary_actors.info()
md_title_principals_reduced.info()


#%% md_PrimaryActorsList - Basic Analysis

md_PrimaryActorsList_2 = md_PrimaryActorsList.drop(["Number", "Name", "nconst", "Ratings", "Rating Years", "tconst", "Films_Training_Qty", "Films_Testing_Qty"], axis=1)
md_PrimaryActorsList_corr = md_PrimaryActorsList_2.corr()
sns.pairplot(md_PrimaryActorsList_2)


#%% md_secondary_actors - Basic Analysis

md_secondary_actors_2 = md_secondary_actors.drop(["name", "nconst", "tconsts"], axis=1)
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

def populate_input_actor_scores_for_film(film_tconst, desired_number_of_primary_actors, desired_number_of_secondary_actors):
    #find every relavent primary actor, as select the ones with the highest film count
    """ find a way to filter according to index"""
    
    
    primary_actors_in_film = md_actor_to_film[md_actor_to_film.index.get_level_values('tconst') == film_tconst]
    primary_actors_in_film['count'] = None
    
    for index in primary_actors_in_film.index:
        row_in_df = md_PrimaryActorsList.loc[md_PrimaryActorsList['Name'] == primary_actors_in_film['actor name'][index]].index[0]
        #primary_actors_in_film['count'][index] = len(md_PrimaryActorsList['tconst'][row_in_df])
        primary_actors_in_film.loc[index, 'count'] = len(md_PrimaryActorsList['tconst'][row_in_df])
    
    primary_actors_to_be_populated_qty = min(desired_number_of_primary_actors, len(primary_actors_in_film))
    actors_to_populate_tconsts = np.array([])
    
    for i in range(0, primary_actors_to_be_populated):
        max_count = primary_actors_in_film['count'].max()
        target_index = primary_actors_in_film.loc[primary_actors_in_film['count'] == max_count].index
        actors_to_populate_tconsts = np.append(actors_to_populate_tconsts, target_index[0][0])
        primary_actors_in_film = primary_actors_in_film.drop(target_index)
        
        
def return_tconsts_in_film_with_highest_film_counts(film_tconst, actors_qty, actor_DB, previous_tconst):
    # called twice by populate_input_actor_scores_for_film
    if len(previous_tconst) = 0:
        tconsts = np.array([])
    else:
        tconsts = previous_tconst
    return tconsts    
    
    
    
    
    
    
    

#filtered_primary_actor_list = print(md_actor_to_film[np.in1d(md_actor_to_film.index.get_level_values(1), [film_tconst])])
film_tconst = 'tt1067106'
populate_input_actor_scores_for_film(film_tconst, 0)

#md_actor_to_film[(nm0001715, tt0118528)]

#print(md_actor_to_film[np.in1d(md_actor_to_film.index.get_level_values(0), [film_tconst])])
