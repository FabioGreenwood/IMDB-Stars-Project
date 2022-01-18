# -*- coding: utf-8 -*-
'''#Config control
0.0.3 - removal of np arrays in df primary actors to allow for the saving of generated metadata to and from csv's 
0.0.4 transferal to spyder

#Required actions:
    remapping of files,
    download of files


'''


"""
Spyder Editor

This is a temporary script file.
"""

import pdb
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Final_Training_Year = 2015

#%%

from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
start = datetime.now()
Final_Training_Year = 2015
df_title_principals = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_principals.tsv", sep='\t')
df_title_ratings = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_ratings.tsv", sep='\t')
df_name_basics = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/name_basics.tsv", sep='\t')
#df_title_akas = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_akas.tsv", sep='\t')
df_title_basics = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_basics.tsv", sep='\t')
#df_title_crew = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/title_crew.tsv", sep='\t')
print("Time taken: " + str(datetime.now() - start))

#%%
#define global variables




#%%

#Re-loading .csv's
df_PrimaryActorsList = pd.read_csv("/content/drive/MyDrive/Data_Analysis_Project/Metadata/df_PrimaryActorsList.csv")
relister_main(df_PrimaryActorsList, 'Ratings')
df_PrimaryActorsList.head()
df_PrimaryActorsList['Ratings'][0]

#%%
# initialise and Reset df_PrimaryActorsList
df_PrimaryActorsList_column_values = ["Number", "Name", "Rel_IDs", "Ratings", "Rating Years", 'Tconts', "Rating - Mean", "Rating - Std Dev", "Start Year", "Final Year", "Films_Training", "Films_Testing"]
df_PrimaryActorsList = pd.DataFrame(columns = df_PrimaryActorsList_column_values)
df_PrimaryActorsList.head()


#%%

#Saving .csv's
df_PrimaryActorsList.to_csv('df_PrimaryActorsList.csv')
!cp df_PrimaryActorsList.csv "drive/My Drive/Data_Analysis_Project/Metadata"


#%%


#Find all the top actors
PrimaryActorsList = ["Jack Nicholson", "Marlon Brando", "Robert De Niro", "Al Pacino", "Daniel Day-Lewis", "Dustin Hoffman", "Tom Hanks", "Anthony Hopkins", "Paul Newman", "Denzel Washington", "Spencer Tracy", "Laurence Olivier", "Jack Lemmon", "Michael Caine", "James Stewart", "Robin Williams", "Robert Duvall", "Sean Penn", "Morgan Freeman", "Jeff Bridges", "Sidney Poitier", "Peter O'Toole", "Clint Eastwood", "Gene Hackman", "Charles Chaplin", "Ben Kingsley", "Philip Seymour Hoffman", "Leonardo DiCaprio", "Russell Crowe", "Kevin Spacey", "Humphrey Bogart", "Gregory Peck", "Clark Gable", "Gary Cooper", "George C. Scott", "Jason Robards", "Charles Laughton", "Anthony Quinn", "Peter Sellers", "James Cagney", "Peter Finch", "Henry Fonda", "Cary Grant", "Richard Burton", "Burt Lancaster", "William Holden", "John Wayne", "Kirk Douglas", "Alec Guinness", "Christopher Plummer", "Tommy Lee Jones", "Sean Connery", "Alan Arkin", "Christopher Walken", "Joe Pesci", "Ian McKellen", "Michael Douglas", "Jon Voight", "Albert Finney", "Geoffrey Rush", "Jeremy Irons", "Javier Bardem", "Heath Ledger", "Christoph Waltz", "Ralph Fiennes", "Johnny Depp", "Benicio Del Toro", "Jamie Foxx", "Joaquin Phoenix", "Colin Firth", "Matthew McConaughey", "Christian Bale", "Gary Oldman", "Edward Norton", "Brad Pitt", "Tom Cruise", "Matt Damon", "Hugh Jackman", "Robert Downey Jr.", "Liam Neeson", "Mel Gibson", "Harrison Ford", "Woody Allen", "Steve McQueen", "Orson Welles", "Robert Redford", "James Dean", "Charlton Heston", "Gene Kelly", "Robert Mitchum", "Bill Murray", "Samuel L. Jackson", "Jim Carrey", "Don Cheadle", "Martin Sheen", "Alan Rickman", "Edward G. Robinson", "Will Smith", "John Goodman", "Buster Keaton"]
df_PrimaryActorsList_column_values_reduced = ["Number", "Name", "Rel_IDs", "Ratings", "Rating Years", 'Tconts']

#reset values
returnedFlag = []
df_PrimaryActorsList = df_PrimaryActorsList[0:0]
row_number = 0
blank_lists  = []

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

#%%
#definition of methods

def relister_main(dataframe, column_name, format='float'):
  for i in range(0, len(dataframe)):
    relister_single(dataframe, column_name, i, format)

def relister_single(dataframe, column_name, row_num, format='float'):
  if len(df_PrimaryActorsList[column_name][row_num]) <= 2:
    return
  if np.isreal(df_PrimaryActorsList[column_name][row_num][0]):
    return
  separator = ','
  target_string = df_PrimaryActorsList[column_name][row_num]
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
    
    if first == True:
      new_list = [value]
      first = False
    else:
      new_list.append(value)
    
  dataframe[column_name][row_num] = new_list

#find all the ratings for an actor's films 
def get_film_ratings_v1 (df_PrimaryActorsList, actor_ID, df_title_basics, df_title_ratings ):
  line_item = df_PrimaryActorsList.loc[df_PrimaryActorsList['Name'] == 'Jack Nicholson']
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
        #pdb.set_trace()
        if len(df_PrimaryActorsList['Ratings'][actor_ID]) == 0:
          df_PrimaryActorsList['Ratings'][actor_ID] = [value.values[0]]
          df_PrimaryActorsList['Rating Years'][actor_ID] = [title_year.values[0]]
          df_PrimaryActorsList['Tconts'][actor_ID] = [title]
        else:
          df_PrimaryActorsList['Ratings'][actor_ID].append(value.values[0])
          df_PrimaryActorsList['Rating Years'][actor_ID].append(title_year.values[0])
          df_PrimaryActorsList['Tconts'][actor_ID].append(title)
          
# calculate actor meta data
def define_actor_metadata_2(datatable, row):
  
  rating_data = datatable['Ratings'][row]
  rating_years_data = datatable['Rating Years'][row]
  
  #error catches
  if not(len(rating_data) == len(rating_years_data)):
    breakpoint() #these values should be the same length

  films_count = len(rating_data)
  training_films_count = 0
  datatable['Rating - Mean'][row] = sum(rating_data) / len(rating_data)
  datatable['Rating - Std Dev'][row] = np.std(rating_data)
  datatable['Start Year'][row] = min(rating_years_data)
  datatable['Final Year'][row] = max(rating_years_data)
  for i in rating_years_data:
    #pdb.set_trace()
    if (float(i) <= Final_Training_Year) :
      training_films_count += 1
  datatable['Films_Training'][row] = training_films_count
  datatable['Films_Testing'][row] = films_count - datatable['Films_Training'][row]





#%%

# development of following method:
def visualise_ratings_career(actor_name_or_ID = 'Jack Nicholson', dataframe = df_PrimaryActorsList):
  #index initialied at impossible value to force failre down the line if the if statments dont catch it
  actor_index = len(dataframe) + 5
  
  #actor_name_or_ID.str
  
  if isinstance(actor_name_or_ID, str):
    actor_index = float(dataframe[dataframe['Name'] == actor_name_or_ID].index)
  elif isinstance(actor_name_or_ID, int):
    actor_index = actor_name_or_ID

  ratings = dataframe['Ratings'][actor_index]
  rating_years = dataframe['Rating Years'][actor_index]
  plt.scatter(rating_years, ratings);
  
  plt.show()


visualise_ratings_career(2)
    
    
    



