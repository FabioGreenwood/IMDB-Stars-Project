# -*- coding: utf-8 -*-
'''#Config control
0.0.3 - removal of np arrays in df primary actors to allow for the saving of generated metadata to and from csv's 
0.0.4 transferal to spyder

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
runcell("Import Modules and CSVs", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
runcell("Regenerate df_PrimaryActorsList", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
runcell("Call Generate Actor Metadata", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')


#%% start up run
runcell("Import Modules and CSVs", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
runcell("Re-loading.csv's", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
runcell("Call Generate Actor Metadata", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')



#%% Call Generate Actor Metadata

'''This is to be consolodated to one'''

#runcell("Generate PrimaryActorsList", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
runcell("Generate Actor Metadata", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
runcell("Generate actor to film metadata", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')

#%% Regenerate df_PrimaryActorsList

print(datetime.now())
runcell("Reset Metatables (df_PrimaryActorsList, md_actor_to_film) and Generate df_PrimaryActorsList", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
print(datetime.now())
get_film_ratings_v1(0)
print(datetime.now())
get_film_ratings_v1(1)
print(datetime.now())
get_film_ratings_v1(2)
print(datetime.now())


#%% Import Modules and CSVs
''' Import modules, data and initialise global variables'''

from datetime import datetime
start = datetime.now()
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

#from google.colab import drive
#drive.mount('/content/drive')
#from google.colab import files
#start = datetime.now()
df_title_principals = pd.read_csv(r"C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project/title_principals.tsv", sep='\t')
df_title_ratings = pd.read_csv(r"C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project/title_ratings.tsv", sep='\t')
df_name_basics = pd.read_csv(r"C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\name_basics.tsv", sep='\t')
#df_title_akas = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_akas.tsv", sep='\t')
df_title_basics = pd.read_csv(r"C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\title_basics.tsv", sep='\t')
#df_title_crew = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_crew.tsv", sep='\t')
#print("Time taken: " + str(datetime.now() - start))


#Generate the metadata columns required in primary tables
#df_title_principals['Relative Scoring'] = False

# General Variables
df_PrimaryActorsList_column_values = ["Number", "Name", "Rel_IDs", "Ratings", "Rating Years", 'Tconts', "Rating - Mean", "Rating - Std Dev", "Start Year", "Final Year", "Films_Training_Qty", "Films_Testing_Qty", "Training Mean", "Training Std", "Model_Gradient", "Model_Intercept", "Model Rating 2020"]
df_PrimaryActorsList = pd.DataFrame(columns = df_PrimaryActorsList_column_values)

df_RatingModels_column_names = ["Name", 'Tconts', "Model"]
df_RatingModels = pd.DataFrame(columns = df_RatingModels_column_names, dtype=object)

md_actor_to_film_column_names = ['tconts', 'nconst', 'film name', 'actor name', 'film year', 'film score', 'actor relative score']
md_actor_to_film = pd.DataFrame(columns = md_actor_to_film_column_names)

md_secondary_actors_column_names = ['ncont', 'Name', 'Start Year', 'Final Year', 'Relative Actor Scores', 'Relative Actor Score - Mean']
md_secondary_actors = pd.DataFrame(columns = md_secondary_actors_column_names)

md_film_scores_column_names = ['tconst', 'name', 'genre', 'Primary Actor Ratings', 'Primary Actor Ratings - Mean', 'Primary Actor Ratings - Std Dev']
md_film_scores = pd.DataFrame(columns = md_film_scores_column_names)

#initialise general methods and global variables
Final_Training_Year = 2005
title_principals_unique_category = []
runcell("General Methods", 'C:/Users/fabio/OneDrive/Documents/Studies/Programming_Analysis_Project/programming_analysis_project_main 0.0.4.py')
get_unique_values__title_principals_category()
roles_of_secondary_interest = ['self', 'actor', 'actress']

print("Time taken: " + str(datetime.now() - start))


#%% Re-loading.csv's

#Re-loading .csv's

df_PrimaryActorsList = pd.read_csv(r"C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\Metadata\df_PrimaryActorsList.csv")
relister_main(df_PrimaryActorsList, 'Ratings')
relister_main(df_PrimaryActorsList, 'Rating Years', "int")
relister_main(df_PrimaryActorsList, 'Tconts', "string")

md_actor_to_film = pd.read_csv(r'C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\Metadata\md_actor_to_film.csv')

md_film_scores = pd.read_csv(r'C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\Metadata\md_film_scores.csv')
relister_main(md_film_scores, 'genre', "string")
relister_main(md_film_scores, 'Primary Actor Ratings', 'float')


#%%

#relister_main(df_PrimaryActorsList, 'Tconts', "string")

#%% Save .csv's 

# Commented out lines are for colab 

#df_PrimaryActorsList.to_csv('df_PrimaryActorsList.csv')
#!cp df_PrimaryActorsList.csv "drive/My Drive/Data_Analysis_Project/Metadata"

df_PrimaryActorsList.to_csv(r'C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\Metadata\df_PrimaryActorsList.csv')
md_actor_to_film.to_csv(r'C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\Metadata\md_actor_to_film.csv')
md_film_scores.to_csv(r'C:\Users\fabio\OneDrive\Documents\Studies\Programming_Analysis_Project\Metadata\md_film_scores.csv')



#%% Reset Metatables (df_PrimaryActorsList, md_actor_to_film) and Generate df_PrimaryActorsList


# initialise and Reset [df_PrimaryActorsList, md_actor_to_film]
df_PrimaryActorsList_column_values = ["Number", "Name", "Rel_IDs", "Ratings", "Rating Years", 'Tconts', "Rating - Mean", "Rating - Std Dev", "Start Year", "Final Year", "Films_Training_Qty", "Films_Testing_Qty", "Training Mean", "Training Std", "Model_Gradient", "Model_Intercept", "Model Rating 2020"]
df_PrimaryActorsList = pd.DataFrame(columns = df_PrimaryActorsList_column_values)

md_actor_to_film_column_names = ['tconts', 'nconst', 'film name', 'actor name','film year', 'film score', 'actor relative score']
md_actor_to_film = pd.DataFrame(columns = md_actor_to_film_column_names)

#Find all the top actors 1
PrimaryActorsList = ["Jack Nicholson", "Marlon Brando", "Robert De Niro", "Al Pacino", "Daniel Day-Lewis", "Dustin Hoffman", "Tom Hanks", "Anthony Hopkins", "Paul Newman", "Denzel Washington", "Spencer Tracy", "Laurence Olivier", "Jack Lemmon", "Michael Caine", "James Stewart", "Robin Williams", "Robert Duvall", "Sean Penn", "Morgan Freeman", "Jeff Bridges", "Sidney Poitier", "Peter O'Toole", "Clint Eastwood", "Gene Hackman", "Charles Chaplin", "Ben Kingsley", "Philip Seymour Hoffman", "Leonardo DiCaprio", "Russell Crowe", "Kevin Spacey", "Humphrey Bogart", "Gregory Peck", "Clark Gable", "Gary Cooper", "George C. Scott", "Jason Robards", "Charles Laughton", "Anthony Quinn", "Peter Sellers", "James Cagney", "Peter Finch", "Henry Fonda", "Cary Grant", "Richard Burton", "Burt Lancaster", "William Holden", "John Wayne", "Kirk Douglas", "Alec Guinness", "Christopher Plummer", "Tommy Lee Jones", "Sean Connery", "Alan Arkin", "Christopher Walken", "Joe Pesci", "Ian McKellen", "Michael Douglas", "Jon Voight", "Albert Finney", "Geoffrey Rush", "Jeremy Irons", "Javier Bardem", "Heath Ledger", "Christoph Waltz", "Ralph Fiennes", "Johnny Depp", "Benicio Del Toro", "Jamie Foxx", "Joaquin Phoenix", "Colin Firth", "Matthew McConaughey", "Christian Bale", "Gary Oldman", "Edward Norton", "Brad Pitt", "Tom Cruise", "Matt Damon", "Hugh Jackman", "Robert Downey Jr.", "Liam Neeson", "Mel Gibson", "Harrison Ford", "Woody Allen", "Steve McQueen", "Orson Welles", "Robert Redford", "James Dean", "Charlton Heston", "Gene Kelly", "Robert Mitchum", "Bill Murray", "Samuel L. Jackson", "Jim Carrey", "Don Cheadle", "Martin Sheen", "Alan Rickman", "Edward G. Robinson", "Will Smith", "John Goodman", "Buster Keaton"]
df_PrimaryActorsList_column_values_reduced = ["Number", "Name", "Rel_IDs", "Ratings", "Rating Years", 'Tconts']

#reset values
returnedFlag = []
df_PrimaryActorsList = df_PrimaryActorsList[0:0]
row_number = 0
blank_lists  = []

#Find all the top actors 2
for i in PrimaryActorsList[:3]:
  row_number += 1
  nameK = str(i)
  rel_key = []
  rel_key_mask = df_name_basics['primaryName'] == i
  #pdb.set_trace()
  rel_key_slice = df_name_basics['nconst'].values
  rel_key = rel_key_slice[rel_key_mask]
  #rel_key = df_name_basics['nconst'][rel_key_mask]
  #pdb.set_trace()
  #rel_key = df_name_basics[df_name_basics['primaryName'] == i].index
  #print(str(row_number), str(nameK), str(rel_key))
  addition = pd.Series(data=[row_number, nameK, rel_key, blank_lists, blank_lists, blank_lists], index=df_PrimaryActorsList_column_values_reduced)
  df_PrimaryActorsList = df_PrimaryActorsList.append(addition, ignore_index=True)
  


df_PrimaryActorsList.head()


#%% Generate Actor Metadata

for i in np.arange(0, len(df_PrimaryActorsList)):
    
    define_actor_metadata_step_1(df_PrimaryActorsList, i)
    M, C, rating2020, Mean, Std = create_actor_model(i, df_PrimaryActorsList)

    df_PrimaryActorsList["Model_Gradient"][i] = float(M)
    df_PrimaryActorsList["Model_Intercept"][i] = float(C)
    df_PrimaryActorsList["Model Rating 2020"][i] = float(rating2020)
    
    df_PrimaryActorsList["Training Mean"][i] = float(Mean)
    df_PrimaryActorsList["Training Std"][i] = float(Std)
    

#%% General Methods
#definition of methods

def relister_main(dataframe, column_name, format='float'):
  for i in range(0, len(dataframe)):
    relister_single(dataframe, column_name, i, format)

def relister_single(dataframe, column_name, row_num, format='float'):
  if len(dataframe[column_name][row_num]) <= 2:
    return
  if np.isreal(dataframe[column_name][row_num][0]):
    return
  separator = ','
  target_string = dataframe[column_name][row_num]
  if target_string[0] == '[':
      target_string = target_string[1:-1]
  new_list_raw = target_string.split(',')
  new_list = []
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
      new_list = [value]
      first = False
    else:
      new_list.append(value)
    
  dataframe[column_name][row_num] = new_list

#find an actor's films and their ratings
def get_film_ratings_v1(actor_ID, df_title_basics=df_title_basics, df_title_ratings=df_title_ratings):
  
  global df_PrimaryActorsList, md_actor_to_film
  
  #reset cells about to be populated
  df_PrimaryActorsList['Ratings'][actor_ID] = []
  df_PrimaryActorsList['Rating Years'][actor_ID] = []
  df_PrimaryActorsList['Tconts'][actor_ID] = []
    
  for ncont_rel in df_PrimaryActorsList['Rel_IDs'][actor_ID]:
    titles_mask = df_title_principals['nconst'] == ncont_rel
    titles = df_title_principals['tconst'][titles_mask]
    
    #pdb.set_trace()
    for title in titles:
      rating_index = df_title_ratings[df_title_ratings['tconst'] == title].index
      if rating_index.size > 0:
        value = df_title_ratings['averageRating'][rating_index] 
        title_year_index  = df_title_basics[df_title_basics['tconst'] == title].index
        title_year = df_title_basics['startYear'][title_year_index]
        
        addition_index = ['tconts', 'nconst', 'film name', 'actor name','film year', 'film score', 'actor relative score']
        addition_2 = pd.Series(data=[title, df_PrimaryActorsList['Rel_IDs'][actor_ID][0], df_title_basics['originalTitle'][title_year_index].values[0], df_PrimaryActorsList['Name'][actor_ID], int(title_year), float(value), None], index=addition_index)
        md_actor_to_film = md_actor_to_film.append(addition_2, ignore_index=True)
        
        
        
        #pdb.set_trace()
        if len(df_PrimaryActorsList['Ratings'][actor_ID]) == 0:
          df_PrimaryActorsList['Ratings'][actor_ID] = [value.values[0]]
          df_PrimaryActorsList['Rating Years'][actor_ID] = [int(title_year.values[0])]
          df_PrimaryActorsList['Tconts'][actor_ID] = [title]
        else:
          df_PrimaryActorsList['Ratings'][actor_ID].append(value.values[0])
          df_PrimaryActorsList['Rating Years'][actor_ID].append(int(title_year.values[0]))
          df_PrimaryActorsList['Tconts'][actor_ID].append(title)
        
        #add relationship to specialist table
        #['tconts', 'nconst', 'film name', 'actor name', 'actor relative score']
        
        
          
          
# calculate actor meta data



def define_actor_metadata_step_1(datatable, row):
    ''' Generates various items of metadata per actor
    Namely: Rating - Mean, Rating - Std Dev, Start Year, Final Year, Films_Training_Qty, Films_Testing_Qty
    first of various actor metadata generation steps'''    
    
    
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
    for i in rating_years_data:
        #pdb.set_trace()
        if (float(i) <= Final_Training_Year) :
            training_films_count += 1
    datatable['Films_Training_Qty'][row] = training_films_count
    datatable['Films_Testing_Qty'][row] = films_count - datatable['Films_Training_Qty'][row]


def visualise_ratings_career(actor_name_or_ID = 'Jack Nicholson', dataframe = df_PrimaryActorsList):
  #FG comment: index initialied at impossible value to force failure down the line if the if statments dont catch it
  actor_index = len(dataframe) + 5
  
  #actor_name_or_ID.str
  
  if isinstance(actor_name_or_ID, str):
    actor_index = dataframe[dataframe['Name'] == actor_name_or_ID].index[0]
  else:
    actor_index = actor_name_or_ID

  ratings = dataframe['Ratings'][actor_index]
  rating_years = dataframe['Rating Years'][actor_index]
  plt.scatter(rating_years, ratings);
  plt.title(dataframe['Name'][actor_index])
  
  plt.show()

def create_actor_model(actor_name_or_ID = 'Jack Nicholson', dataframe = df_PrimaryActorsList, final_training_year = Final_Training_Year, generate_chart=False, print_chart = False):
    
    '''   # This method splits the training data of an actor and 
    returns the linear regression model, std dev and charts the prediction 
    if required '''

    
    
    #FG comment: index initialied at impossible value to force failure down the line if the if statments dont catch it
    actor_index = len(dataframe) + 5
    
    #actor_name_or_ID.str
    
    if isinstance(actor_name_or_ID, str):
      actor_index = dataframe[dataframe['Name'] == actor_name_or_ID].index[0]
    else:
      actor_index = actor_name_or_ID
    
    
    ratings = dataframe['Ratings'][actor_index]
    rating_years = dataframe['Rating Years'][actor_index]
    
    training_mask = [(film < final_training_year + 1) for film in rating_years]
    #training_mask = list(map(lambda x: x <= final_training_year, rating_years))
    
    ratings_training = [ratings[i] for i in range(len(ratings)) if training_mask[i]]
    rating_years_training = [rating_years[i] for i in range(len(rating_years)) if training_mask[i]]
    
    ratings_testing = [ratings[i] for i in range(len(ratings)) if not(training_mask[i])]
    rating_years_testing = [rating_years[i] for i in range(len(rating_years)) if not(training_mask[i])]
        
    ratings = np.array(ratings).reshape(-1, 1)
    rating_years = np.array(rating_years).reshape(-1, 1)
          
    model = LinearRegression().fit(rating_years, ratings)
    #print(model.score(rating_years, ratings))
    #print(model.predict(np.array(2000).reshape(1, -1)))
    training_mean = np.mean(ratings_testing)
    training_std = np.std(ratings_testing)
    rating2020 = model.predict(np.array(2020).reshape(1,-1))[0][0]
    
    if generate_chart == True:
        plt.scatter(rating_years_training, ratings_training, label="Training")
        plt.scatter(rating_years_testing, ratings_testing, label="Testing")
        x_model = np.arange(min(rating_years), max(rating_years)+1).reshape(-1,1)
        plt.plot(x_model, model.predict(x_model), color="Red", linewidth=3, linestyle="--")
        
        #plt.scatter(rating_years_testing, ratings_testing, label="Testing")
        
        plt.title(dataframe['Name'][actor_index] + " Model Display")
        plt.show()
        if print_chart == True:
            return model.coef_[0][0], model.intercept_[0][0], rating2020, training_mean, training_std, plt
    
    return model.coef_[0], model.intercept_[0], rating2020, training_mean, training_std

def get_unique_values__title_principals_category():
    global title_principals_unique_category #list of unique roles in a film
    column_values = df_title_principals[["category"]].values.ravel()
    title_principals_unique_category =  pd.unique(column_values)
    
def get_unique_values__relevant_film_tconts():
    unique_tconts_ = df_PrimaryActorsList["Tconts"].values.ravel()
    unique_tconts = []
    for sublist in unique_tconts_:
        for item in sublist:
            unique_tconts.append(item)
        
    unique_tconts = pd.unique(unique_tconts)
    #unique_tconts = list(unique_tconts[1:-1].split(","))
    #for i in range(0, len(unique_tconts)):
    #    unique_tconts[i] = unique_tconts[i].replace("'","")
    #    unique_tconts[i] = unique_tconts[i].replace(" ","")
    return unique_tconts

def get_unique_values__primary_actor_nconts():
    unique_primary_nconts_ = []
    '''for i in range(0,len(df_PrimaryActorsList["Rel_IDs"])):
        if i == 0:
            unique_primary_nconts_ = df_PrimaryActorsList["Rel_IDs"][i]
        else:
            unique_primary_nconts_ = np.concatenate((unique_primary_nconts_, df_PrimaryActorsList["Rel_IDs"][i]))'''
    unique_primary_nconts_ = df_PrimaryActorsList["Rel_IDs"].values.ravel()
    unique_primary_nconts = []
    for sublist in unique_primary_nconts_:
        for item in sublist:
            unique_primary_nconts.append(item)
    
    return unique_primary_nconts

def evaluate_actor_to_film_score(actor_expected_2020_score, actor_gradient, actor_std_dev, film_year, film_score):
    year_of_intercept = 2020
    actor_expected_score_for_films_year = (film_year - year_of_intercept) * actor_gradient + actor_expected_2020_score
    relative_z_difference = (film_score - actor_expected_score_for_films_year) / actor_std_dev
    return relative_z_difference

def evaluate_film_relative_scores(md_film_scores=md_film_scores, df_PrimaryActorsList = df_PrimaryActorsList):
        
    return md_film_scores

#%% Generate actor to film metadata




actor_name = ""
for i in range(0, len(md_actor_to_film)):
    actor_name_new = md_actor_to_film['actor name'][i]
    if not(actor_name == actor_name_new):
        actor_name = actor_name_new
        actor_index = df_PrimaryActorsList[df_PrimaryActorsList['Name'] == actor_name].index[0]
        actor_expected_2020_score   = df_PrimaryActorsList['Model Rating 2020'][actor_index]
        actor_gradient              = df_PrimaryActorsList['Model_Gradient'][actor_index]
        actor_std_dev               = df_PrimaryActorsList['Training Std'][actor_index]
    film_year   = md_actor_to_film['film year'][i]
    film_score  = md_actor_to_film['film score'][i]
    
    md_actor_to_film['actor relative score'][i] = float(evaluate_actor_to_film_score(actor_expected_2020_score, actor_gradient, actor_std_dev, film_year, film_score))



#%% Pre_pop md_film_scores

unique_tconts = get_unique_values__relevant_film_tconts()

md_film_scores_column_names = ['tconst', 'name', 'genre', 'Primary Actor Ratings', 'Primary Actor Ratings - Mean', 'Primary Actor Ratings - Std Dev']
md_film_scores = pd.DataFrame(columns = md_film_scores_column_names)

md_film_scores = md_film_scores[0:0]
for i in range(0, len(unique_tconts)):
    df_index = df_title_basics[df_title_basics['tconst'] == unique_tconts[i]].index
    data = pd.Series([unique_tconts[i], df_title_basics['primaryTitle'][df_index].values[0], df_title_basics['genres'][df_index].values[0], [], None, None], index=md_film_scores_column_names)
    md_film_scores = md_film_scores.append(data, ignore_index=True)


#%% Reduce df_title_principals and collect Secondary Actor Metadata

#get_unique_values__title_principals_category()

unique_primary_nconts = get_unique_values__primary_actor_nconts()

list_categories = ['self', 'actor', 'actress']
mask_for_category = [True if ele in list_categories else False for ele in df_title_principals['category']]
df_title_principals_reduced = df_title_principals[mask_for_category]

unique_tconts = get_unique_values__relevant_film_tconts()
mask_for_relevent_film = [True if ele in unique_tconts else False for ele in df_title_principals_reduced['tconst']]
df_title_principals_reduced = df_title_principals_reduced[mask_for_relevent_film]

df_title_principals_reduced['Relative Scoring'] = False
df_title_principals_reduced['Relationship Class'] = 0

#%%


#mask_for_actor = df_title_principals['category'] == 
#mask_for_actress = df_title_principals['nconst'] == 
#mask_for_self = df_title_principals['nconst'] == 




