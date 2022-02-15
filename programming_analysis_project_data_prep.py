# -*- coding: utf-8 -*-
'''#Config control
0.0.3 - removal of np arrays in df primary actors to allow for the saving of generated metadata to and from csv's 
0.0.4 transferal to spyder and continued work
0.0.5 Re-orginisation of code
0.0.6 Incremental save
0.0.7 Adjustment of the method of spliting the training set from according to a cut-off from 

#Required actions:
    remapping of files,
    download of files


'''
''' Dev notes
1. Every instance of this: actor_index = dataframe[dataframe['Name'] == actor_name_or_ID].index[0] needs to be potentially replaced


'''


"""
Spyder Editor

This is a temporary script file.
"""
#%% Hard restart/regeneration of data


option_to_generate_training_set = False

runcell("Import Modules and CSVs", script_filepath + dataprep_filename)
print("1")
print(datetime.now())
runcell("Reset and Define Metatables and General Variables/Methods", script_filepath + dataprep_filename)
print("2")
print(datetime.now())
runcell("General Methods", script_filepath + dataprep_filename)
print("3")
print(datetime.now())
runcell("md_PrimaryActorsList - Step 1", script_filepath + dataprep_filename)
print("4")
print(datetime.now())
runcell("md_PrimaryActorsList - Step 2", script_filepath + dataprep_filename)
print("5")
print(datetime.now())
runcell("Generate Actor Metadata", script_filepath + dataprep_filename)
print("6")
print(datetime.now())
runcell("Generate actor to film metadata", script_filepath + dataprep_filename)
print("7")
print(datetime.now())
runcell("md_film_scores - Step 1", script_filepath + dataprep_filename)
print("8")
print(datetime.now())
runcell("md_film_scores - Step 2", script_filepath + dataprep_filename)
print("9")
print(datetime.now())
runcell("collect Secondary Actor Metadata - Step 1", script_filepath + dataprep_filename)
print("10")
print(datetime.now())
runcell("collect Secondary Actor Metadata - Step 2", script_filepath + dataprep_filename)
print("11")
print(datetime.now())
runcell("collect Secondary Actor Metadata - Step 3", script_filepath + dataprep_filename)
print(datetime.now())
runcell("collect Secondary Actor Metadata - Step 4", script_filepath + dataprep_filename)



#%%
    






#%% start up run

script_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\"
database_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\data\\"
metadata_filepath = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\Metadata\\"

dataprep_filename = "programming_analysis_project_data_prep.py"
analysis_filename = "programming_analysis_project_analysis_file.py"

runcell("Import Modules and CSVs", script_filepath + dataprep_filename)
runcell("Reset and Define Metatables and General Variables/Methods", script_filepath + dataprep_filename)
runcell("General Methods", script_filepath + dataprep_filename)
runcell("Re-loading.csv's", script_filepath + dataprep_filename)

#%% Save progress 

runcell("Save .csv's", script_filepath + dataprep_filename)


#%% Import Modules and CSVs
''' Import modules, data and initialise global variables'''

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





#from google.colab import drive
#drive.mount('/content/drive')
#from google.colab import files
#start = datetime.now()
df_title_principals = pd.read_csv(database_filepath + "title_principals.tsv", sep='\t')
df_title_ratings = pd.read_csv(database_filepath + "title_ratings.tsv", sep='\t')
df_title_ratings.set_index('tconst', inplace=True)

df_name_basics = pd.read_csv(database_filepath + "name_basics.tsv", sep='\t')
df_name_basics.set_index('nconst', inplace=True)
#df_name_basics.set_index('nconst', inplace=True)
#df_title_akas = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_akas.tsv", sep='\t')
df_title_basics = pd.read_csv(database_filepath + "title_basics.tsv", sep='\t')
df_title_basics.set_index('tconst', inplace=True)
#relister_main(df_title_basics, 'genres', "string")
#df_title_basics.set_index('tconst', inplace=True)
#df_title_crew = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_crew.tsv", sep='\t')
#print("Time taken: " + str(datetime.now() - start))





#%% Re-loading.csv's

#Re-loading .csv's



md_actor_to_film = pd.read_csv(metadata_filepath + 'md_actor_to_film.csv')
md_actor_to_film.set_index(['nconst', 'tconst'], inplace=True)
md_actor_to_film = md_actor_to_film.loc[:, ~md_actor_to_film.columns.str.contains('^Unnamed')]


md_actor_to_film_secondary = pd.read_csv(metadata_filepath + 'md_actor_to_film_secondary.csv')
md_actor_to_film_secondary.set_index(['nconst', 'tconst'], inplace=True)
md_actor_to_film_secondary = md_actor_to_film_secondary.loc[:, ~md_actor_to_film_secondary.columns.str.contains('^Unnamed')]


md_film_scores = pd.read_csv(metadata_filepath + 'md_film_scores.csv')
relister_main(md_film_scores, 'genre', "string")
relister_main(md_film_scores, 'Primary Actor Relative Ratings', 'float')
relister_main(md_film_scores, 'nconst', 'string')
md_film_scores = md_film_scores.loc[:, ~md_film_scores.columns.str.contains('^Unnamed')]
md_film_scores.set_index('tconst', inplace=True)


md_PrimaryActorsList = pd.read_csv(metadata_filepath + "md_PrimaryActorsList.csv")
relister_main(md_PrimaryActorsList, 'Ratings')
relister_main(md_PrimaryActorsList, 'Rating Years', "int")
relister_main(md_PrimaryActorsList, 'tconst', "string")
relister_main(md_PrimaryActorsList, 'nconst', "string")


md_secondary_actors = pd.read_csv(metadata_filepath + 'md_secondary_actors.csv')
relister_main(md_secondary_actors, 'tconst', "string")
relister_main(md_secondary_actors, 'Relative Actor Scores', 'float')
relister_main(md_secondary_actors, 'Film Years', 'int')
relister_main(md_secondary_actors, 'nconst', 'string')
#md_secondary_actors = md_secondary_actors.loc[:, ~md_secondary_actors.columns.str.contains('^Unnamed')]
md_secondary_actors = md_secondary_actors.loc[:, ~md_secondary_actors.columns.str.contains('^Unnamed')]

md_title_principals_reduced = pd.read_csv(metadata_filepath + 'md_title_principals_reduced.csv')




#%% Save .csv's 

# Commented out lines are for colab 

#md_PrimaryActorsList.to_csv('md_PrimaryActorsList.csv')
#!cp md_PrimaryActorsList.csv "drive/My Drive/Data_Analysis_Project/Metadata"


md_actor_to_film.to_csv(metadata_filepath + 'md_actor_to_film.csv')
md_actor_to_film_secondary.to_csv(metadata_filepath + 'md_actor_to_film_secondary.csv')
md_film_scores.to_csv(metadata_filepath + 'md_film_scores.csv')
md_PrimaryActorsList.to_csv(metadata_filepath + 'md_PrimaryActorsList.csv')
md_secondary_actors.to_csv(metadata_filepath + 'md_secondary_actors.csv')

md_title_principals_reduced.to_csv(metadata_filepath + 'md_title_principals_reduced.csv')




#%% Reset and Define Metatables and General Variables/Methods

#Reset and Define Metatables 
md_PrimaryActorsList_column_values = ["Number", "Name", "nconst", "Ratings", "Rating Years", 'tconst', "Rating - Mean", "Rating - Std Dev", "Start Year", "Final Year", "Films_Training_Qty", "Films_Testing_Qty", "Model_Gradient", "Model_Intercept", "Model Rating 2020"]
md_PrimaryActorsList = pd.DataFrame(columns = md_PrimaryActorsList_column_values)

md_RatingModels_column_names = ["Name", 'tconst', "Model"]
md_RatingModels = pd.DataFrame(columns = md_RatingModels_column_names, dtype=object)

md_actor_to_film_column_names = ['tconst', 'nconst', 'film name', 'actor name','film year', 'film score', 'actor relative score']
md_actor_to_film = pd.DataFrame(columns = md_actor_to_film_column_names)
md_actor_to_film.set_index(['nconst', 'tconst'], inplace=True)


md_actor_to_film_secondary = pd.DataFrame(columns = md_actor_to_film_column_names)
md_actor_to_film_secondary.set_index(['nconst', 'tconst'], inplace=True)

md_secondary_actors_column_names = ['name', 'nconst', 'Start Year', 'Final Year', 'tconst', 'Relative Actor Scores', 'Film Years', 'Relative Actor Score - Mean', "Relative Actor Score - Std", "Count"]
md_secondary_actors = pd.DataFrame(columns = md_secondary_actors_column_names)
#md_secondary_actors.set_index('name', inplace=True)

md_film_scores_column_names = ['tconst', 'name', 'genre', 'film year', 'rating', 'nconst', 'Primary Actor Relative Ratings', 'Primary Actor Relative Ratings - Mean', 'Primary Actor Relative Ratings - Std Dev']
md_film_scores = pd.DataFrame(columns = md_film_scores_column_names)#, index=['tconst'])
md_film_scores.set_index('tconst', inplace=True)

PrimaryActorsList = ["Jack Nicholson", "Marlon Brando", "Robert De Niro", "Al Pacino", "Daniel Day-Lewis", "Dustin Hoffman", "Tom Hanks", "Anthony Hopkins", "Paul Newman", "Denzel Washington", "Spencer Tracy", "Laurence Olivier", "Jack Lemmon", "Michael Caine", "James Stewart", "Robin Williams", "Robert Duvall", "Sean Penn", "Morgan Freeman", "Jeff Bridges", "Sidney Poitier", "Peter O'Toole", "Clint Eastwood", "Gene Hackman", "Charles Chaplin", "Ben Kingsley", "Philip Seymour Hoffman", "Leonardo DiCaprio", "Russell Crowe", "Kevin Spacey", "Humphrey Bogart", "Gregory Peck", "Clark Gable", "Gary Cooper", "George C. Scott", "Jason Robards", "Charles Laughton", "Anthony Quinn", "Peter Sellers", "James Cagney", "Peter Finch", "Henry Fonda", "Cary Grant", "Richard Burton", "Burt Lancaster", "William Holden", "John Wayne", "Kirk Douglas", "Alec Guinness", "Christopher Plummer", "Tommy Lee Jones", "Sean Connery", "Alan Arkin", "Christopher Walken", "Joe Pesci", "Ian McKellen", "Michael Douglas", "Jon Voight", "Albert Finney", "Geoffrey Rush", "Jeremy Irons", "Javier Bardem", "Heath Ledger", "Christoph Waltz", "Ralph Fiennes", "Johnny Depp", "Benicio Del Toro", "Jamie Foxx", "Joaquin Phoenix", "Colin Firth", "Matthew McConaughey", "Christian Bale", "Gary Oldman", "Edward Norton", "Brad Pitt", "Tom Cruise", "Matt Damon", "Hugh Jackman", "Robert Downey Jr.", "Liam Neeson", "Mel Gibson", "Harrison Ford", "Woody Allen", "Steve McQueen", "Orson Welles", "Robert Redford", "James Dean", "Charlton Heston", "Gene Kelly", "Robert Mitchum", "Bill Murray", "Samuel L. Jackson", "Jim Carrey", "Don Cheadle", "Martin Sheen", "Alan Rickman", "Edward G. Robinson", "Will Smith", "John Goodman", "Buster Keaton"]
md_PrimaryActorsList_column_values_reduced = ["Number", "Name", "nconst", "Ratings", "Rating Years", 'tconst']




#General Variables/Methods
#Final_Training_Year = 2005
title_principals_unique_category = list([])
runcell("General Methods", script_filepath + dataprep_filename)
get_unique_values__title_principals_category()
roles_of_secondary_interest = ['self', 'actor', 'actress']

#%% md_PrimaryActorsList - Step 1

#reset values
returnedFlag = list([])
md_PrimaryActorsList = md_PrimaryActorsList[0:0]
row_number = 0
blank_lists  = list([])

#Find all the top actors 2
for i in PrimaryActorsList:
#for i in PrimaryActorsList:
  row_number += 1
  nameK = str(i)
  rel_key = list([])
  rel_key_mask = df_name_basics['primaryName'] == i
  #pdb.set_trace()
  rel_key_slice = df_name_basics.index
  rel_key = rel_key_slice[rel_key_mask]
  #rel_key = df_name_basics['nconst'][rel_key_mask]
  #pdb.set_trace()
  #rel_key = df_name_basics[df_name_basics['primaryName'] == i].index
  #print(str(row_number), str(nameK), str(rel_key))
  addition = pd.Series(data=[row_number, nameK, list(rel_key), blank_lists, blank_lists, blank_lists], index=md_PrimaryActorsList_column_values_reduced)
  md_PrimaryActorsList = md_PrimaryActorsList.append(addition, ignore_index=True)
  


#%% md_PrimaryActorsList - Step 2


if option_to_generate_training_set == True:
    md_title_principals_reduced, training_tconsts, testing_tconsts = filter_testing_tconsts_from_title_principles(unique_tconsts, md_title_principals_reduced)



def md_PrimaryActorsList(training_set = df_title_principals):
    #changing
    counter = 0
    for i in md_PrimaryActorsList.index:
        counter += 1    
        if len(md_PrimaryActorsList.index) < 10:
            print(datetime.now())
        elif float(counter) % int(len(md_PrimaryActorsList.index) / 10) == 0:
            print("-----")
            print(datetime.now())
            PC = float(float(i) / len(md_PrimaryActorsList.index))
            print(str(i) + " / " + str(md_PrimaryActorsList.index))
            print(PC)
    
        #runcell("Reset Metatables (md_PrimaryActorsList, md_actor_to_film) and Generate md_PrimaryActorsList", script_filepath + dataprep_filename)
        get_film_ratings_v1(i, df_title_basics, df_title_ratings, training_set)
    
    print(md_PrimaryActorsList['name'][md_PrimaryActorsList.index[94]])
    #get_film_ratings_v1(md_PrimaryActorsList.index[96], df_title_basics, df_title_ratings, training_set)
    #get_film_ratings_v1(md_PrimaryActorsList.index[97], df_title_basics, df_title_ratings, training_set)
    #get_film_ratings_v1(md_PrimaryActorsList.index[98], df_title_basics, df_title_ratings, training_set)
    #get_film_ratings_v1(md_PrimaryActorsList.index[99], df_title_basics, df_title_ratings, training_set)

    
md_PrimaryActorsList()



#%%

get_film_ratings_v1(md_PrimaryActorsList.index[95])

#%% Generate Actor Metadata

for i in np.arange(0, len(md_PrimaryActorsList)):
    
    define_actor_metadata_step_1(md_PrimaryActorsList, i)
    M, C, rating2020 = create_actor_model(i, md_PrimaryActorsList)

    md_PrimaryActorsList["Model_Gradient"][i] = float(M)
    md_PrimaryActorsList["Model_Intercept"][i] = float(C)
    md_PrimaryActorsList["Model Rating 2020"][i] = float(rating2020)
    
    



#%% Generate actor to film metadata

actor_name = ""
for i in md_actor_to_film.index:
    
    actor_name_new = md_actor_to_film['actor name'][i]
    
    if not type(actor_name_new) == str:
        actor_name_new = actor_name_new[0]
        
    if not(actor_name == actor_name_new):
        actor_name = actor_name_new
        actor_index = md_PrimaryActorsList[md_PrimaryActorsList['name'] == actor_name].index[0]
        actor_expected_2020_score   = md_PrimaryActorsList['Model Rating 2020'][actor_index]
        actor_gradient              = md_PrimaryActorsList['Model_Gradient'][actor_index]
        actor_std_dev               = md_PrimaryActorsList['Rating - Std Dev'][actor_index]
    film_year   = md_actor_to_film['film year'][i]
    film_score  = md_actor_to_film['film score'][i]
    
    if film_year.size > 1:
        film_year = film_year[0]
        film_score = film_score[0]
    
    
    md_actor_to_film['actor relative score'][i] = float(evaluate_actor_to_film_score(actor_expected_2020_score, actor_gradient, actor_std_dev, film_year, film_score))



#%% md_film_scores - Step 1

unique_tconst = get_unique_values__relevant_film_tconst()

md_film_scores = md_film_scores[0:0]

for df_index in unique_tconst :#range(0, len(unique_tconst)):

    #df_index = df_title_basics[df_title_basics['tconst'] == unique_tconst[i]].index
    #data = pd.Series([unique_tconst[i], df_title_basics['primaryTitle'][df_index].values[0], df_title_basics['genres'][df_index].values[0], list([]), None, None], index=md_film_scores_column_names[1:])
    data = pd.Series(data=[df_title_basics['primaryTitle'][df_index], df_title_basics['genres'][df_index].split(","), df_title_basics['startYear'][df_index], float(df_title_ratings['averageRating'][df_index]), list([]), list([]), None, None], index=md_film_scores_column_names[1:], name=df_index)
    md_film_scores = md_film_scores.append(data)

'''
df_title_basics['primaryTitle'][df_index].values[0], 
df_title_basics['genres'][df_index].values[0],
df_title_basics['startYear'][df_index], 
list([]), list([]), None, None], index=md_film_scores_column_names[1:]
'''




#%% md_film_scores - Step 2


n=0

for film_rel_index in md_actor_to_film.index:
   
    target_nconst = film_rel_index[0]
    target_tconst = film_rel_index[1]
    
    target_rating = md_actor_to_film['actor relative score'][film_rel_index][0]
    
    md_film_scores.loc[target_tconst, 'nconst'].append(target_nconst)
    md_film_scores.loc[target_tconst, 'Primary Actor Relative Ratings'].append(target_rating)
    
for film_index in md_film_scores.index:
    md_film_scores.loc[film_index, 'Primary Actor Relative Ratings - Mean'] = np.mean(md_film_scores.loc[film_index, 'Primary Actor Relative Ratings'])
    md_film_scores.loc[film_index, 'Primary Actor Relative Ratings - Std Dev'] = np.std(md_film_scores.loc[film_index, 'Primary Actor Relative Ratings'])







#%% Generate md_title_principals_reduced

#list_categories = ['self', 'actor', 'actress']

md_title_principals_reduced_training, training_tconsts, testing_tconsts = filter_testing_tconsts_from_title_principles(unique_tconsts, md_title_principals_reduced)

unique_tconst = get_unique_values__relevant_film_tconst()
mask_for_film = [True if ele in unique_tconst else False for ele in df_title_principals['tconst']]
md_title_principals_reduced = df_title_principals[mask_for_film]

list_categories = ['self', 'actor', 'actress']
mask_for_category = [True if ele in list_categories else False for ele in md_title_principals_reduced['category']]
md_title_principals_reduced = md_title_principals_reduced[mask_for_category]


unique_secondary_nconst = get_unique_values(md_title_principals_reduced, 'nconst')

#%% Generate md_title_principals_reduced - alturnative With hardcoded counter






print(datetime.now())
mask_for_film_x = [True if ele in unique_tconst else False for ele in df_title_principals['tconst'][:10000000]]
df_x = pd.DataFrame(mask_for_film_x)
df_x.to_csv(metadata_filepath + 'mask_for_film_1.csv')
print(datetime.now())
print(len(mask_for_film_x))
print("1")


mask_for_film_x = mask_for_film_x + [True if ele in unique_tconst else False for ele in df_title_principals['tconst'][10000000:20000000]]
df_x = pd.DataFrame(mask_for_film_x)
df_x.to_csv(metadata_filepath + 'mask_for_film_2.csv')
print(datetime.now())
print(len(mask_for_film_x))
print("2")

mask_for_film_x = mask_for_film_x + [True if ele in unique_tconst else False for ele in df_title_principals['tconst'][20000000:30000000]]
df_x = pd.DataFrame(mask_for_film_x)
df_x.to_csv(metadata_filepath + 'mask_for_film_3.csv')
print(datetime.now())
print("3")

mask_for_film_x = mask_for_film_x + [True if ele in unique_tconst else False for ele in df_title_principals['tconst'][30000000:40000000]]
df_x = pd.DataFrame(mask_for_film_x)
df_x.to_csv(metadata_filepath + 'mask_for_film_4.csv')
print(datetime.now())
print("4")

mask_for_film_x = mask_for_film_x + [True if ele in unique_tconst else False for ele in df_title_principals['tconst'][40000000:]]
df_x = pd.DataFrame(mask_for_film_x)
df_x.to_csv(metadata_filepath + 'mask_for_film_5.csv')
print(datetime.now())
print("5")

md_title_principals_reduced = df_title_principals[mask_for_film_x]



#%% collect Secondary Actor Metadata - Step 1

unique_secondary_nconst = get_unique_values(md_title_principals_reduced, 'nconst')
md_secondary_actors = md_secondary_actors[0:0]

#md_secondary_actors = md_secondary_actors.loc[:, ~md_secondary_actors.columns.str.contains('^Unnamed')]
#md_secondary_actors = md_secondary_actors.loc[:, ~md_secondary_actors.columns.str.contains('0')]

counter = 0
columns = md_secondary_actors_column_names[0:2]
#Collect all associated names to nconst (to later group according to name)
for nconst in unique_secondary_nconst:
    
    try:
        name = df_name_basics.loc[nconst, 'primaryName']
    except KeyError as err:
        name = nconst
    
    data=[name, nconst]#, Start_Year, Final_Year, tconst, Relative_Actor_Scores, film_years, Relative_Actor_Score_Mean]
    
    
        
    addition = pd.Series(data=data, index=columns)
    md_secondary_actors = md_secondary_actors.append(addition, ignore_index=True)
    
    
    #counter
    counter += 1
    if float(counter) % int(len(unique_secondary_nconst) / 10) == 0:
        print("-----")
        print(datetime.now())
        PC = float(float(counter) / len(unique_secondary_nconst))
        print(str(counter ) + " / " + str(len(unique_secondary_nconst)))
        print(PC)
    #counter
    
    
   
    
#md_secondary_actors_2 = pd.DataFrame(md_secondary_actors.groupby('name')['nconst'].apply(list))

md_secondary_actors["Start Year"]                   = None
md_secondary_actors["Final Year"]                   = None
md_secondary_actors["tconst"]                      = None
md_secondary_actors["Relative Actor Scores"]        = None
md_secondary_actors["Film Years"]                   = None
md_secondary_actors["Relative Actor Score - Mean"]  = None
md_secondary_actors["Relative Actor Score - Std"]   = None

counter = 0
for i in range(0, len(md_secondary_actors)):
    md_secondary_actors["tconst"][i]               = list([])
    md_secondary_actors["Relative Actor Scores"][i] = list([])
    md_secondary_actors["Film Years"][i]            = list([])
    
    #counter
    counter += 1
    if float(counter) % int(len(md_secondary_actors) / 10) == 0:
        print("-----")
        print(datetime.now())
        PC = float(float(counter) / len(md_secondary_actors))
        print(str(counter ) + " / " + str(len(md_secondary_actors)))
        print(PC)
    #counter
    
    

#%% collect Secondary Actor Metadata - Step 2

print(datetime.now())
iterations = len(md_title_principals_reduced)


#reset values 
for n in range(0, len(md_secondary_actors)):
    md_secondary_actors['tconst'][n]                       = list([])
    md_secondary_actors['Relative Actor Scores'][n]         = list([])
    md_secondary_actors['Film Years'][n]                    = list([])
    md_secondary_actors['Relative Actor Score - Mean'][n]   = None
    md_secondary_actors['Relative Actor Score - Std'][n]    = None


#%% collect Secondary Actor Metadata - Step 3
print(datetime.now())
counter = 0
start = datetime.now()
for i in md_title_principals_reduced.index:
    #progress timer
    #if i % int(iterations / 10) == 0:
    #    PC = float(i / iterations)
    #    print(str(i) + " / " + str(iterations))
    #    print(PC)
    #body of method
    nconst = md_title_principals_reduced['nconst'][i]
    tconst = md_title_principals_reduced['tconst'][i]
    found = False
    actor_index = 0
    
    while found == False and actor_index <= len(md_secondary_actors):
        
        if nconst in md_secondary_actors['nconst'][actor_index]:
            found = True
        else:
            actor_index += 1
    
    #film_index = md_film_scores[md_film_scores['tconst'] == tconst].index[0]
    md_secondary_actors['tconst'][actor_index].append(tconst)
    md_secondary_actors['Relative Actor Scores'][actor_index].append(md_film_scores['Primary Actor Relative Ratings - Mean'][tconst])
    md_secondary_actors['Film Years'][actor_index].append(int(md_film_scores['film year'][tconst]))
    
    #counter
    counter += 1
    if float(counter) % int(len(md_title_principals_reduced) / 100) == 0:
        print("-----")
        print(datetime.now())
        PC = float(float(counter) / len(md_title_principals_reduced))
        print(str(counter ) + " / " + str(len(md_title_principals_reduced)))
        print(PC)
        print("end estimate @: " + str(((datetime.now() - start) / 0.5) + datetime.now()))
    #counter
    
    
md_secondary_actors['Count'] = ""
print(datetime.now())    
for x in range(0,len(md_secondary_actors)):
    md_secondary_actors['Count'][x] = len(md_secondary_actors['Relative Actor Scores'][x])    
    if len(md_secondary_actors['Relative Actor Scores'][x]) > 0:
        md_secondary_actors['Relative Actor Score - Mean'][x] = np.mean(md_secondary_actors['Relative Actor Scores'][x])
        md_secondary_actors['Relative Actor Score - Std'][x] = np.std(md_secondary_actors['Relative Actor Scores'][x])
        
        md_secondary_actors['Start Year'][x] = min(md_secondary_actors['Film Years'][x])
        md_secondary_actors['Final Year'][x] = max(md_secondary_actors['Film Years'][x])
        
    

print(datetime.now())

#%% collect Secondary Actor Metadata - Step 4

md_secondary_actors['Count'] = 0
for x in range(0,len(md_secondary_actors)):
    md_secondary_actors['Count'][x] = len(md_secondary_actors['Relative Actor Scores'][x])

md_actor_to_film_secondary  = return_populated_md_actor_to_film_secondary_dataframe(md_secondary_actors, md_actor_to_film_column_names)

    
#md_PrimaryActorsList[md_PrimaryActorsList['Name'] == actor_name].index[0]

#index = md_secondary_actors.loc[md_secondary_actors['nconst'].str.contains("nm0583948", case=False)]


#rating_index = md_secondary_actors[md_secondary_actors['tconst'].contains('nm0583948')].index

#md_secondary_actors.set_index('tconst')
#df_test = md_secondary_actors.groupby('name')['nconst'].apply(list)





#%% Rejected code
for nconst in unique_secondary_nconst:
    name = df_name_basics.loc[nconst, 'primaryName']
    Start_Year = None
    Final_Year = None
    tconst = list([])
    Relative_Actor_Scores = list([])
    film_years = list([])
    Relative_Actor_Score_Mean = None
    
    data=[name, Start_Year, Final_Year, tconst, Relative_Actor_Scores, film_years, Relative_Actor_Score_Mean]
    name=nconst
    column=md_secondary_actors_column_names[1:]
    addition_2 = pd.Series(data=data, name=name, column=column)
    

'''
md_secondary_actors_column_names = ['nconst', 'name', 'Start Year', 'Final Year', 'Relative Actor Scores', 'Relative Actor Score - Mean']


addition_index = md_actor_to_film.columns
addition_2 = pd.Series(data=[title, md_PrimaryActorsList['nconst'][actor_ID][0], df_title_basics['originalTitle'][title_year_index].values[0], md_PrimaryActorsList['name'][actor_ID], int(title_year), float(value), None], index=addition_index)
md_actor_to_film = md_actor_to_film.append(addition_2, ignore_index=True)

'''


#%% REJECTED CODE Reduce df_title_principals and collect Secondary Actor Metadata

'''XXXXXXXXXXXXXXXXXXXXXXX I'm not sure this is still needed'''

#get_unique_values__title_principals_category()

unique_primary_nconts = get_unique_values__primary_actor_nconts()

list_categories = ['self', 'actor', 'actress']
mask_for_category = [True if ele in list_categories else False for ele in df_title_principals['category']]
df_title_principals_reduced = df_title_principals[mask_for_category]

unique_tconst = get_unique_values__relevant_film_tconst()
mask_for_relevent_film = [True if ele in unique_tconst else False for ele in df_title_principals_reduced['tconst']]
df_title_principals_reduced = df_title_principals_reduced[mask_for_relevent_film]

df_title_principals_reduced['Relative Scoring'] = False
df_title_principals_reduced['Relationship Class'] = 0

#%% 




#%% Split Metatable Function

def split_meta_table(table, training_tconst, testing_tconst, columns_to_copy, columns_to_split, split_index = 'tconst'):
    #this function splits various tables into to separate testing and training tables according to which tconst values have been deemed to be training films or testing films
    #it will then populate the data on films deemed for training in the table_training and data on the films deemed for table_testing and return both tables
    #please note values such as 'start year', 'final year', mean and standard deviations will have to be repopulated after this function
    table_training = table[0:0]
    table_testing = table[0:0]
    
    for target_column in columns_to_copy:
        if not target_column in table.columns:
            raise Exception("Error column named" + target_column + "not in table targeted for spliting")
    
    for target_column in columns_to_split:
        if not target_column in table.columns:
            raise Exception("Error column named" + target_column + "not in table targeted for spliting")
        
    
    for table_index in table.index:
        
        table_training = table_training.append(pd.Series(data=None, name=table_index))
        table_testing = table_testing.append(pd.Series(data=None, name=table_index))
        tconst = table[split_index][table_index]
        
        for target_column in columns_to_copy:
            table_training[target_column][table_index] = table[target_column][table_index]
            table_testing[target_column][table_index] = table[target_column][table_index]
        
        
        for target_column in columns_to_split:
            table_training[target_column][table_index] = list([])
            table_testing[target_column][table_index] = list([])
            for t in range(0, len(tconst)):
                if tconst[t] in training_tconst:
                    table_training[target_column][table_index].append(table[target_column][table_index][t])
                    #table_training[target_column][table_index] = table_training[target_column][table_index].append(table[target_column][table_index][t])
                elif tconst[t] in testing_tconst:
                    table_testing[target_column][table_index].append(table[target_column][table_index][t])
                    #table_testing[target_column][table_index] = table_testing[target_column][table_index].append(table[target_column][table_index][t])
                else:
                    raise Exception("Error tconst: " + tconst[t] + " not sepcified as either for the training or testing dataset, check inputs")
            
        return table_training, table_testing
        
    

columns_to_copy = ['name', 'nconst']
columns_to_split = ['tconst', 'Ratings']

table_training, table_testing = split_meta_table(md_PrimaryActorsList, unique_tconst[:5], unique_tconst[5:], columns_to_copy, columns_to_split)




#%% General Methods
#definition of methods

def relister_main(dataframe, column_name, format='float'):
  for i in range(0, len(dataframe)):
    relister_single(dataframe, column_name, i, format)

def relister_single(dataframe, column_name, row_num, format='float', return_value = False):
  if len(dataframe[column_name][row_num]) <= 2:
    dataframe[column_name][row_num] = list([])
    return
  if np.isreal(dataframe[column_name][row_num][0]):
    return
  separator = ','
  target_string = dataframe[column_name][row_num]
  if target_string[0] == '[':
      target_string = target_string[1:-1]
  new_list_raw = target_string.split(',')
  new_list = list([])
  first = True
  for i in new_list_raw:
    if format=='float':
      value = float(i)
    elif format=='int':
        string = i.strip()
        string = string.replace("'","")
        value = int(string)
    elif format=='string':
        string = i.strip()
        string = string.replace("'","")
        value = string
    
    if first == True:
      new_list = list([value])
      first = False
    else:
      new_list.append(value)
      
  if return_value == False:
      dataframe[column_name][row_num] = new_list
  else:
      return new_list

def relister_single_functional(cell, format_input):
  if len(cell) <= 2:
    cell = list([])
    return
  if np.isreal(cell[0]):
    return
  separator = ','
  target_string = cell
  if target_string[0] == '[':
      target_string = target_string[1:-1]
  new_list_raw = target_string.split(',')
  new_list = list([])
  first = True
  for i in new_list_raw:
    if format_input=='float':
      value = float(i)
    elif format_input=='int':
        string = i.strip()
        string = string.replace("'","")
        value = int(string)
    elif format_input=='string':
        string = i.strip()
        string = string.replace("'","")
        value = string
    
    if first == True:
      new_list = list([value])
      first = False
    else:
      new_list.append(value)
      
  return new_list

#find an actor's films and their ratings
def get_film_ratings_v1(actor_ID, df_title_basics, df_title_ratings, df_title_principals):
  
  global md_PrimaryActorsList, md_actor_to_film
  
  #reset cells about to be populated
  md_PrimaryActorsList['Ratings'][actor_ID] = list([])
  md_PrimaryActorsList['Rating Years'][actor_ID] = list([])
  md_PrimaryActorsList['tconst'][actor_ID] = list([])
    
  for ncont_rel in md_PrimaryActorsList['nconst'][actor_ID]:
    titles_mask = df_title_principals['nconst'] == ncont_rel
    titles = df_title_principals['tconst'][titles_mask]
    
    #pdb.set_trace()
    for title in titles:
      #rating_index = df_title_ratings[df_title_ratings['tconst'] == title].index
      if title in df_title_ratings.index:
        value = df_title_ratings['averageRating'][title] 
        
        title_year = df_title_basics['startYear'][title]
        
        addition_index = md_actor_to_film.columns
        addition_2 = pd.Series(data=[df_title_basics['originalTitle'][title], md_PrimaryActorsList['name'][actor_ID], int(title_year), float(value), None], name=(md_PrimaryActorsList['nconst'][actor_ID][0], title) , index=addition_index)
        md_actor_to_film = md_actor_to_film.append(addition_2)
        
        
        
        #pdb.set_trace()
        if len(md_PrimaryActorsList['Ratings'][actor_ID]) == 0:
          md_PrimaryActorsList['Ratings'][actor_ID] = [value]
          #md_PrimaryActorsList['Ratings'][actor_ID] = [value.values[0]]
          md_PrimaryActorsList['Rating Years'][actor_ID] = [title_year]
          md_PrimaryActorsList['tconst'][actor_ID] = [title]
        else:
          md_PrimaryActorsList['Ratings'][actor_ID].append(value)
          #md_PrimaryActorsList['Ratings'][actor_ID].append(value.values[0])
          md_PrimaryActorsList['Rating Years'][actor_ID].append(title_year)
          md_PrimaryActorsList['tconst'][actor_ID].append(title)
        
        #add relationship to specialist table
        #['tconst', 'nconst', 'film name', 'actor name', 'actor relative score']
        
        
          
          
# calculate actor meta data



def define_actor_metadata_step_1(datatable, row):
    ''' Generates various items of metadata per actor
    Namely: Rating - Mean, Rating - Std Dev, Start Year, Final Year, Films_Training_Qty, Films_Testing_Qty
    first of various actor metadata generation steps'''    
    
    datatable['Rating Years'][row] = [int(i) for i in datatable['Rating Years'][row]]

    rating_data = datatable['Ratings'][row]
    rating_years_data = datatable['Rating Years'][row]
          
          #error catches
    if not(len(rating_data) == len(rating_years_data)):
        breakpoint() #these values should be the same length
        
    films_count = len(rating_data)
    training_films_count = 0
    datatable['Rating - Mean'][row] = sum(rating_data) / len(rating_data)
    datatable['Rating - Std Dev'][row] = np.std(rating_data)
    #datatable['Films_Training_Qty'][row] = min(rating_years_data)
    #datatable['Films_Testing_Qty'][row] = max(rating_years_data)
    datatable['Start Year'][row] = min(rating_years_data)
    datatable['Final Year'][row] = max(rating_years_data)
    
    datatable['Films_Training_Qty'][row] = training_films_count
    datatable['Films_Testing_Qty'][row] = films_count - datatable['Films_Training_Qty'][row]


def visualise_ratings_career(actor_name_or_ID = 'Jack Nicholson', dataframe = md_PrimaryActorsList):
  #FG comment: index initialied at impossible value to force failure down the line if the if statments dont catch it
  actor_index = len(dataframe) + 5
  
  #actor_name_or_ID.str
  
  if isinstance(actor_name_or_ID, str):
    actor_index = dataframe[dataframe['name'] == actor_name_or_ID].index[0]
  else:
    actor_index = actor_name_or_ID

  ratings = dataframe['Ratings'][actor_index]
  rating_years = dataframe['Rating Years'][actor_index]
  plt.scatter(rating_years, ratings);
  plt.title(dataframe['name'][actor_index])
  
  plt.show()

def create_actor_model(actor_name_or_ID = 'Jack Nicholson', dataframe = md_PrimaryActorsList, generate_chart=False, print_chart = False):
    
    '''   # This method splits the training data of an actor and 
    returns the linear regression model, std dev and charts the prediction 
    if required '''

    
    
    #FG comment: index initialied at impossible value to force failure down the line if the if statments dont catch it
    actor_index = len(dataframe) + 5
    
    #actor_name_or_ID.str
    
    if isinstance(actor_name_or_ID, str):
      actor_index = dataframe[dataframe['name'] == actor_name_or_ID].index[0]
    else:
      actor_index = actor_name_or_ID
    
    
    ratings = dataframe['Ratings'][actor_index]
    rating_years = dataframe['Rating Years'][actor_index]
    
    #training_mask = [(film < final_training_year + 1) for film in rating_years]
    #training_mask = list(map(lambda x: x <= final_training_year, rating_years))
    
    #ratings_training = [ratings[i] for i in range(len(ratings)) if training_mask[i]]
    #rating_years_training = [rating_years[i] for i in range(len(rating_years)) if training_mask[i]]
    
    #ratings_testing = [ratings[i] for i in range(len(ratings)) if not(training_mask[i])]
    #rating_years_testing = [rating_years[i] for i in range(len(rating_years)) if not(training_mask[i])]
        
    ratings = np.array(ratings).reshape(-1, 1)
    rating_years = np.array(rating_years).reshape(-1, 1)
          
    model = LinearRegression().fit(rating_years, ratings)
    #print(model.score(rating_years, ratings))
    #print(model.predict(np.array(2000).reshape(1, -1)))
    #training_mean = np.mean(ratings_testing)
    #training_std = np.std(ratings_testing)
    rating2020 = model.predict(np.array(2020).reshape(1,-1))[0][0]
    
    if generate_chart == True:
        plt.scatter(rating_years_training, ratings_training, label="Training")
        plt.scatter(rating_years_testing, ratings_testing, label="Testing")
        x_model = np.arange(min(rating_years), max(rating_years)+1).reshape(-1,1)
        plt.plot(x_model, model.predict(x_model), color="Red", linewidth=3, linestyle="--")
        
        #plt.scatter(rating_years_testing, ratings_testing, label="Testing")
        
        plt.title(dataframe['name'][actor_index] + " Model Display")
        plt.show()
        if print_chart == True:
            return model.coef_[0][0], model.intercept_[0][0], rating2020, training_mean, training_std, plt
    
    return model.coef_[0], model.intercept_[0], rating2020#, training_mean, training_std

def get_unique_values__title_principals_category():
    global title_principals_unique_category #list of unique roles in a film
    column_values = df_title_principals[["category"]].values.ravel()
    title_principals_unique_category =  pd.unique(column_values)
    
def get_unique_values__relevant_film_tconst():
    unique_tconst_ = md_PrimaryActorsList["tconst"].values.ravel()
    unique_tconst = list([])
    for sublist in unique_tconst_:
        for item in sublist:
            unique_tconst.append(item)
        
    unique_tconst = pd.unique(unique_tconst)
    #unique_tconst = list(unique_tconst[1:-1].split(","))
    #for i in range(0, len(unique_tconst)):
    #    unique_tconst[i] = unique_tconst[i].replace("'","")
    #    unique_tconst[i] = unique_tconst[i].replace(" ","")
    return unique_tconst

def get_unique_values__primary_actor_nconts():
    unique_primary_nconts_ = list([])
    '''for i in range(0,len(md_PrimaryActorsList["nconst"])):
        if i == 0:
            unique_primary_nconts_ = md_PrimaryActorsList["nconst"][i]
        else:
            unique_primary_nconts_ = np.concatenate((unique_primary_nconts_, md_PrimaryActorsList["nconst"][i]))'''
    unique_primary_nconts_ = md_PrimaryActorsList["nconst"].values.ravel()
    unique_primary_nconts = list([])
    for sublist in unique_primary_nconts_:
        for item in sublist:
            unique_primary_nconts.append(item)
    
    return unique_primary_nconts

def get_unique_values(dataframe, column, single_or_nested='single'):
    unique_values = list([])
    unique_values_temp = dataframe[column].values.ravel()
    if single_or_nested == 'single':
        return unique_values_temp
    else:
        for sublist in unique_values_temp:
            for item in sublist:
                unique_values.append(item)    
        return unique_values


def evaluate_actor_to_film_score(actor_expected_2020_score, actor_gradient, actor_std_dev, film_year, film_score):
    year_of_intercept = 2020
    actor_expected_score_for_films_year = (film_year - year_of_intercept) * actor_gradient + actor_expected_2020_score
    relative_z_difference = (film_score - actor_expected_score_for_films_year) / actor_std_dev
    return relative_z_difference

def evaluate_film_relative_scores(md_film_scores=md_film_scores, md_PrimaryActorsList = md_PrimaryActorsList):
        
    return md_film_scores

def return_populated_md_actor_to_film_secondary_dataframe(md_secondary_actors, md_actor_to_film_column_names = ['tconst', 'nconst', 'film name', 'actor name','film year', 'film score', 'actor relative score']):
    
    
    md_actor_to_film_secondary = pd.DataFrame(columns = md_actor_to_film_column_names)
    md_actor_to_film_secondary.set_index(['nconst', 'tconst'], inplace=True)
        
    unique_tconsts = set(md_actor_to_film.index.get_level_values('tconst'))

    start = datetime.now()

    for md_secondary_actors_row in range(0, len(md_secondary_actors)):
        
        if float(md_secondary_actors_row) % int(len(md_secondary_actors) / 10) == 0:
            print("-----")
            print(datetime.now())
            PC = float(float(md_secondary_actors_row) / len(md_PrimaryActorsList.index))
            print(str(md_secondary_actors_row) + " / " + str(len(md_secondary_actors)))
            print(PC)
            
        
        for film_num in range(0, len(md_secondary_actors['tconst'][md_secondary_actors_row]) ):
            
            film_tconst = md_secondary_actors['tconst'][md_secondary_actors_row][film_num]
            data = [ md_film_scores['name'][film_tconst], md_secondary_actors['name'][md_secondary_actors_row], md_secondary_actors['Film Years'][md_secondary_actors_row][film_num], md_film_scores['rating'][film_tconst], None]
            index = md_actor_to_film_column_names[2:]
            name_1 =str(md_secondary_actors['nconst'][md_secondary_actors_row][0]) 
            name = (name_1 , film_tconst)
            
            addition = pd.Series(data = data, index = index, name = name)
            md_actor_to_film_secondary = md_actor_to_film_secondary.append(addition, ignore_index=False)
    
    return md_actor_to_film_secondary 

def retrive_film_rating(database, film_tconst):
    return database['Rating'][film_tconst]

def fg_counter(counter_value, total_iterations_qty, number_of_updates_required_for_total_run = 10):
    if float(counter_value) % int(total_iterations_qty / number_of_updates_required_for_total_run) == 0:
        print("-----")
        print(datetime.now())
        PC = float(float(counter_value) / total_iterations_qty)
        print(str(counter_value) + " / " + str(total_iterations_qty))
        print(PC)

def filter_testing_tconsts_from_title_principles(unique_tconsts, title_principles_DB):
    '''
    Returns:
        Filtered film to actor relational database (a.k.a "XX_title_principles_XX") - For use in generation of metadata without films from testing set
        List of training tconsts - Informs users on what films are considered within the training set
        List of testing tconsts - Testing set, model will need to be tested from these films
        
    '''
    counter = 0
    training_tconsts, testing_tconsts = train_test_split(unique_tconsts, test_size = 0.2, random_state = 0)
    
    title_principles_DB_temp = title_principles_DB.copy()
    
    for i in range(len(title_principles_DB_temp) - 1, -1, -1):
        
        
        fg_counter(counter, len(title_principles_DB_temp), 1000)
        counter += 1
        
        if title_principles_DB_temp['tconst'][i] in testing_tconsts:
            title_principles_DB_temp.drop(df_title_principals_training.index[i])
    
    return title_principles_DB_temp, training_tconsts, testing_tconsts


#%%




  
#%%



   
    
#%% 
print(relister_single_functional(md_actor_to_film_secondary.index[0][0], 'string'))
 
  
#md_secondary_actors.loc[md_secondary_actors['tconst'] == primary_actors_in_film['actor name'][index]].index[0]