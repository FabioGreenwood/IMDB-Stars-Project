# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:49:08 2022

@author: fabio
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import os
from pathlib import Path
import spyder_kernels
import cachetools
#import functions

from datetime import datetime
import pdb

import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import seaborn as sns
import plotly.express as px
import pathlib
import os
import sklearn
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import pickle


# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

#general formatting
st.set_page_config(
     page_title="IMDB Dud Supporting Stars Study",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

#%% #General Methods and Variables

PrimaryActorsList = ["Jack Nicholson", "Marlon Brando", "Robert De Niro", "Al Pacino", "Daniel Day-Lewis", "Dustin Hoffman", "Tom Hanks", "Anthony Hopkins", "Paul Newman", "Denzel Washington", "Spencer Tracy", "Laurence Olivier", "Jack Lemmon", "Michael Caine", "James Stewart", "Robin Williams", "Robert Duvall", "Sean Penn", "Morgan Freeman", "Jeff Bridges", "Sidney Poitier", "Peter O'Toole", "Clint Eastwood", "Gene Hackman", "Charles Chaplin", "Ben Kingsley", "Philip Seymour Hoffman", "Leonardo DiCaprio", "Russell Crowe", "Kevin Spacey", "Humphrey Bogart", "Gregory Peck", "Clark Gable", "Gary Cooper", "George C. Scott", "Jason Robards", "Charles Laughton", "Anthony Quinn", "Peter Sellers", "James Cagney", "Peter Finch", "Henry Fonda", "Cary Grant", "Richard Burton", "Burt Lancaster", "William Holden", "John Wayne", "Kirk Douglas", "Alec Guinness", "Christopher Plummer", "Tommy Lee Jones", "Sean Connery", "Alan Arkin", "Christopher Walken", "Joe Pesci", "Ian McKellen", "Michael Douglas", "Jon Voight", "Albert Finney", "Geoffrey Rush", "Jeremy Irons", "Javier Bardem", "Heath Ledger", "Christoph Waltz", "Ralph Fiennes", "Johnny Depp", "Benicio Del Toro", "Jamie Foxx", "Joaquin Phoenix", "Colin Firth", "Matthew McConaughey", "Christian Bale", "Gary Oldman", "Edward Norton", "Brad Pitt", "Tom Cruise", "Matt Damon", "Hugh Jackman", "Robert Downey Jr.", "Liam Neeson", "Mel Gibson", "Harrison Ford", "Woody Allen", "Steve McQueen", "Orson Welles", "Robert Redford", "James Dean", "Charlton Heston", "Gene Kelly", "Robert Mitchum", "Bill Murray", "Samuel L. Jackson", "Jim Carrey", "Don Cheadle", "Martin Sheen", "Alan Rickman", "Edward G. Robinson", "Will Smith", "John Goodman", "Buster Keaton"]
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

def visualise_ratings_career(actor_name_or_ID, dataframe, add_linear_regression, include_name_in_chart):
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
  if include_name_in_chart == True:
      plt.title(dataframe['name'][actor_index])
  
  if add_linear_regression == True:
      print("Hello")
      gradient, intercept, rating_2020 = create_actor_model(actor_name_or_ID, dataframe, False, False)
      x_regression = range(min(dataframe['Rating Years'][actor_index]), max(dataframe['Rating Years'][actor_index]))
      plt.plot(x_regression, gradient*x_regression + intercept, color = "red", linewidth=1.6)

  
  plt.show()



def create_actor_model(actor_name_or_ID, dataframe, generate_chart, print_chart):
    
    '''   # This method splits the training data of an actor and 
    returns the linear regression model, std dev and charts the prediction 
    if required '''

    actor_index = len(dataframe) + 5
    
    if isinstance(actor_name_or_ID, str):
      actor_index = dataframe[dataframe['name'] == actor_name_or_ID].index[0]
    else:
      actor_index = actor_name_or_ID
    
    
    ratings = dataframe['Ratings'][actor_index]
    rating_years = dataframe['Rating Years'][actor_index]
    
    ratings = np.array(ratings).reshape(-1, 1)
    rating_years = np.array(rating_years).reshape(-1, 1)
          
    model = LinearRegression().fit(rating_years, ratings)
    rating2020 = model.predict(np.array(2020).reshape(1,-1))[0][0]
    
    return model.coef_[0][0], model.intercept_[0], rating2020

def list_actors_films(actor_name, md_PrimaryActorsList, md_film_scores_tconst_index):
    actor_tconsts = md_PrimaryActorsList.loc[md_PrimaryActorsList['name'] == actor_name]['tconst']
    a = 0
    list_of_film_names = np.array([])
    target = md_PrimaryActorsList.loc[md_PrimaryActorsList['name'] == actor_name]['tconst'][0]
    for i in range(0, len(target)):
        try:
            #new_name = md_film_scores_tconst_index['name'][target[i]]
            new_name = target[i]
            new_name = md_film_scores_tconst_index['name'][new_name]
            list_of_film_names = np.append(list_of_film_names, new_name)
        except TypeError as err:
            a += 1

        
        
    
    return "Function pending" #list_of_film_names
    
    
    

#%% Call in data




#script_directory = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\"
#database_directory = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\data\\"
#metadata_directory = "C:\\Users\\fabio\\OneDrive\\Documents\\Studies\\Programming_Analysis_Project\\IMDB-Stars-Project\\Metadata\\"
streamlit_script_name = "stream_lit_script.py"



@st.cache
def get_data_from_excel(a):
    # md_PrimaryActorsList import
    md_PrimaryActorsList = pd.read_csv(".\\Metadata\\md_PrimaryActorsList.csv")
    relister_main(md_PrimaryActorsList, 'Ratings')
    relister_main(md_PrimaryActorsList, 'Rating Years', "int")
    relister_main(md_PrimaryActorsList, 'tconst', "string")
    relister_main(md_PrimaryActorsList, 'nconst', "string")
    md_PrimaryActorsList = md_PrimaryActorsList.loc[:, ~md_PrimaryActorsList.columns.str.contains('^Unnamed')]
    
    #data re-arragement
    md_PrimaryActorsList_corr       = md_PrimaryActorsList.drop(["Number", "name", "nconst", "Ratings", "Rating Years", "tconst", "Films_Training_Qty"], axis=1)
    md_PrimaryActorsList_dataframe  = md_PrimaryActorsList.drop(["Number", "nconst", "Ratings", "Rating Years", "tconst", "Films_Training_Qty"], axis=1)
    
    
    return md_PrimaryActorsList_dataframe, md_PrimaryActorsList_corr, md_PrimaryActorsList
    #return sns.pairplot(md_PrimaryActorsList)

@st.cache
def get_film_info(b):
    #md_film_scores_tconst_index = pd.read_csv(metadata_directory + 'md_film_scores.csv')
    md_film_scores_tconst_index = pd.read_csv('.\\Metadata\\md_film_scores.csv')
    md_film_scores_tconst_index = md_film_scores_tconst_index.set_index(['tconst'], inplace=True)
    return md_film_scores_tconst_index



#md_PrimaryActorsList, md_PrimaryActorsList_corr, md_PrimaryActorsList_sns = get_data_from_excel()
md_PrimaryActorsList_dataframe, md_PrimaryActorsList_corr, md_PrimaryActorsList_full = get_data_from_excel("b")
md_film_scores_tconst_index = get_film_info("e")




#Pictoral Assets
placeholder_diagram = Image.open(".\\Images\\Place Holder Image.png")
nick_cage_image =                   Image.open(".\\Images\\nick_cage.jpg")
md_PrimaryActorsList_sns_image =    Image.open(".\\Images\\md_PrimaryActorList.jpg")
md_secondary_actors_metadata_image = Image.open(".\\Images\\md_secondary_actors_metadata_image.png")
md_secondary_actors_least_20_films_relative_rating_above_below_05 = Image.open(".\\Images\\md_secondary_actors_least_20_films_relative_rating_above_below_05.png")
md_secondary_actors_least_20_films_relative_rating_above_below_05_Mean_STD_only = Image.open(".\\Images\\md_secondary_actors_least_20_films_relative_rating_above_below_05_Mean_STD_only.png")
FG_data_schema_image = Image.open(".\\Images\\FG_data_schema.png")
IMDB_data_schema_image = Image.open(".\\Images\\IMDB_data_schema.png")
FG_acting_career = Image.open(".\\Images\\FGs acting career.jpg")
rosie_o_donnell = Image.open(".\\Images\\rosie o'donnell.jpg")


add_selectbox = st.sidebar.write("""
         Contents
         - Intro
         - Data Overview            
         - Metadata Generation Overview
         - Primary Actor Film Metadata Analysis
         - Metadata Generation Overview
         - Primary Actor Film Metadata Analysis
         - Primary Actor Film Rating Analysis
         - Secondary Actor Film Metadata Analysis
         - Forecasting Introduction
         - The Million Dollar Question
         - Modelling Difficulties



         
         """
         )
add_selectbox = st.sidebar.write("")
add_selectbox = st.sidebar.write("")
add_selectbox = st.sidebar.write("""
         Nomenclaure
         - nconst
             - Actor unique identifier
             (n -> name)
         - tconst
             - Film/series unique identifier
             (t -> title)
         - Actor Relative Score
             - see "Metadata Generation Overview" section
         """
         )











#%% #Layout Block - Intro
st.title("IMDB Dud Supporting Stars Study")
col1, col2, col3 = st.columns(3)
col2.image(nick_cage_image, caption="Example of divisive actor (personally a fan)")


st.write("""
         This study set out to understand if there were actors that, if hired would worsen a film’s rating?
         
         This is a subjective question and therefore must be clearly defined:
             
         *"Are there supporting actors, who tend to appear in films that underperform given the star power of their leads?"*
         
         """
         )
st.write("")

#Layout Block - Data Intro
st.title("Data Overview")
st.write("This is an **introduction** to the problem")



col1, col2 = st.columns(2)

col1.header("IMDB Native Data")
col1.image(FG_data_schema_image, use_column_width=True)
col1.write("IMDB offered relational DB linking the unique ID of films professional (aka 'name constants' or 'nconsts') to various film unique IDs (aka 'title constants' or 'tconsts').")
col1.write("There were separate tables linking these unique tconsts and nconsts to more descriptive information such as film name, actor name, film ratings, release year, etc")

col2.header("Generated 'Metadata'")
col2.image(IMDB_data_schema_image, use_column_width=True)
col2.write("To undertake the analysis metadata was required to be derived from these relational databases")

st.text("")
st.text("")
st.text("")







#%% #Layout Block - Metadata Generation
st.title("Metadata Generation Overview")
st.subheader("Stage 1")
st.write("""
         To do this the top 100 actors (male and female) were selected (as defined by IMDB.com, link below). 
         These actors here on are referred to as "Primary Actors". All their IMDB ratings were collected. 
         From this a linear regression (over time) for each actor was created, creating an expected film rating for each year.
         
         References:
             
             https://www.imdb.com/interfaces/
             
             https://www.imdb.com/list/ls050274118/
         
         """
         )
st.write("")
st.subheader("Stage 2")
st.write("""
         IMDB's film database was then filtered to just the films containing these Primary Actors. 
         All individuals linked to these films (after a filtering of non-actors) were then listed as "Secondary Actors", 
         who are the eventual focus of this study.
         """
         )
st.write("")
st.subheader("Stage 3")
st.write("""
         For each film a secondary actor was in, their "Relative Actor Score” was collected, the exact definition is detailed below. It is a measure comparing a film's rating to the expected rating for the primary actor (for that year), if there were multiple primary actors then the mean of these values would be assigned to the secondary actor for their involvement in the film.
         
         These scores would be collected for each actor, over the films filtered as described above to produce the following scores for each secondary actor:
         - Relative Actor Score - Mean
         - Relative Actor Score - Std (Standard Deviation)

         """
         )
with st.expander("Relative Actor Score - Mean (and Std) Definition"):
     st.write("""
           Relative Actor Score, is the measure of a film's performance relative to the primary actor's expected performance.
           Say a film is rated 6/10, for the year of the film *"primary actor A"* should be expected to star in a 7/10 rated film and his ratings tend to have a standard deviation of 0.75.
           Then the "Primary Actor Relative Ratings" for that primary actor is calculated as:
        """
        )
     st.write("""
               PARR = (Film Rating - Actor Expected Rating) / Actor Rating Standard Deviation
    """
    )
     st.write("""
               PARR =  (6 - 7) / 0.75 = -1.33
               """
               )
     st.write("""
           This is the value relative is a single primary actor. If there are multiple primary actors on the film then the average of this is taken as the films "Primary Actor Relative Ratings - Mean"
           These values is taken by each secondary actor and analysed for calculating a secondary actors statisitics:
        - Relative Actor Score - Mean
        - Relative Actor Score - Std
           """
         )

st.write("")
st.subheader("Filtering out of “testing set” films")
st.write("""
         After stage 2 was completed all the films (represented by their unique identifiers the “tconst”) where split into two lists, to create a training set of films and a testing set of films. Stage 2 was then repeated, however only considering films from the training set.
         
         This was done to ensure that for the forecast model, the metadata for the primary actors (film ratings, means and standard deviations) and secondary actors (relative film rating) were not swayed by films in the testing set.
         
         See cell: “Generate md_title_principals_reduced” in the file […..project_data.py].
         
         This was done as it would be deemed harmful if the model considered secondary actors whose meta scores were from one film (the film being predicted). Therefore when testing was undertaken, the metadata used as an input would only consider films from the training set.
         """
         )



st.text("")
st.text("")
st.text("")






#%% #Layout Block - Initial Data Exploration
st.title("Metadata Generation Overview")
st.header("IMDB Native Data")
st.write("Given the limited number of features we are able to directly tie to individual actors the following studies where undertaken:")

st.write("""
         - Primary Actor Film Metadata Analysis
             - Here we look at the generated metadata of a primary actor
         - Primary Actor Film Rating Analysis
             - Here we look at individual primary actor's film scores to sense check our data and potentially gain insight onto the behaviour of our sole scoring metric the IMDB average rating
         - Secondary Actor Film Metadata Analysis
             - Here we take the whole cohort of circa #### secondary actors and see if we can spot any distinctions between the higher performing secondary actors and the lower performing ones
         """
         )
st.text("")
st.text("")
st.text("")         
#Layout Block - Primary Actor Film Metadata Analysis
st.title("Primary Actor Film Metadata Analysis")
st.write("Intro to section: ")
st.dataframe(md_PrimaryActorsList_dataframe)
with st.expander("Column Definitions"):
     st.write("""
           - Name - Primary name used by the actor
           - Rating - Mean - The average IMDB rating the actor has achieved over their career
           - Rating - Std Dev - The standard devivation across all IMDB film/series ratings the actor has achieved
           - Start Year - The year of the actors first (registered) title
           - Final Year - The year of the actors final (registered) title
           - Film Count - The number of films registered to the actor
           - Model_Gradient - When applying a linear regression of the actors ratings, the gradient of the line of best fit
           - Model Rating 2020 - When applying a linear regression of an actors ratings, the expected rating for a film/series at 2020
           *(line of best fit's intercept at 2020)*
           
           Not listed *(array values collected from the IMDB original relational DBs, the raw values below where used to derive the above values)*
           - nconst - every unique identifier for an individual associated with the actors name
           - tconst - film unique identifier linking actor to film
           - Rating - each IMDB rating for a film/series
           - Rating Year - the film year the film/series was first release
 
     """)
     
st.write("This is a visualisation")
col1, col2 = st.columns(2)


col1.image(md_PrimaryActorsList_sns_image, use_column_width=True)


#%% #Layout Block - Primary Actor Film Rating Analysis
st.title("Primary Actor Film Rating Analysis")
st.write("Here individual primary actors were examined to see how their indiviual film ratings acted over time. On average a primary actors ratings tended to go up in time (with an average gradient of 0.0072, simular to Marlon Brando), however as there was a high standard deviation in this gradient (0.019) this was definitely not a rule")
st.write("")

col1, col2, col3, col4 = st.columns(4)

col1.subheader("Jack Nicholson")
plot1 = visualise_ratings_career("Jack Nicholson", md_PrimaryActorsList_full , True, False)
st.set_option('deprecation.showPyplotGlobalUse', False)
col1.pyplot(plot1)

#grayscale = placeholder_diagram.convert('LA')
col2.subheader("Marlon Brando")
plot2 = visualise_ratings_career("Marlon Brando", md_PrimaryActorsList_full , True, False)
col2.pyplot(plot2)

col3.subheader("Robert De Niro")
plot3 = visualise_ratings_career("Robert De Niro", md_PrimaryActorsList_full , True, False)
col3.pyplot(plot3)


st.write("Robert De Niro is an example of one of the exceptions mentioned eariler (data pictured above)")
st.write("While the other examples tended to drift up, De Niro's earlier films start with a high average/low spread and over time develop a mid average/high spread")
st.write("Reviewing the *tconsts* (film unique identifiers), it is seen that his earlier films were mainly the mafia classics (godfather, goodfellas), foundly remembered films")
st.write("Where as later in his career RD has attempted projects from a wide range types, to varying degrees of success, such as 'The Irishman' 7.8 rating, 'Meet the Fockers' 6.3 and 'Dirty Grandpa' 5.9")




st.subheader("Primary Actor Film Rating Analysis - Custom View")

col1, col2, col3, = st.columns(3)
option = col1.selectbox(
     'How would you like to be contacted?',
     PrimaryActorsList)

col1, col2, col3, = st.columns(3)
#https://stackoverflow.com/questions/69578431/how-to-fix-streamlitapiexception-expected-bytes-got-a-int-object-conver
md_transposed = md_PrimaryActorsList_dataframe.loc[md_PrimaryActorsList_dataframe['name'] == option].T
test = md_transposed.astype(str)
col1.dataframe(test)

plot_custom = visualise_ratings_career(option, md_PrimaryActorsList_full , True, True)
col2.pyplot(plot_custom)
#film_list = list_actors_films(option, md_PrimaryActorsList_full,md_film_scores_tconst_index)
#col3.dataframe(film_list )


st.text("")
st.text("")
st.text("")


#%% #Layout Block - Metadata Generation


st.title("Secondary Actor Film Metadata Analysis")

st.write("Intro to section")
st.write("An initial examination of the secondary actor metadata doesn't reveal much information, partly to the small number of extractable features from the dataset")
col1, col2 = st.columns(2)
col1.image(md_secondary_actors_metadata_image, use_column_width=True)
with col1.expander("Column Definitions"):
     st.write("""
           - Start Year - The year of the actors first (registered) title
           - Final Year - The year of the actors final (registered) title
           
           - Count - The number of films registered to the actor
           
           
           Not listed *(array values collected from the IMDB original relational DBs, the raw values below where used to derive the above values)*
           - nconst - every unique identifier for an individual associated with the actors name
           - tconst - film unique identifier linking actor to film
           
           - Relative Actor Score - Mean - The 
           - Relative Actor Score - Std (Standard Deviation) - each IMDB rating for a film/series
           
 
     """)

st.write("However filtering the secondary actors clears up the data a little:")
col1, col2 = st.columns(2)
col1.image(md_secondary_actors_least_20_films_relative_rating_above_below_05 , use_column_width=True)
st.write("Here the red group show actors with at least 20 appearances and a 'Relative Actor Score - Mean' score of 0.5 or above")
st.write("Whereas the blue group also have at least 20 appearances but a 'Relative Actor Score - Mean' score of -0.5 or below")
with st.expander("Relative Actor Score - Mean (and Std) Definition"):
     st.write("""
           Relative Actor Score, is the measure of a film's performance relative to the primary actor's expected performance.
           Say a film is rated 6/10, for the year of the film *"primary actor A"* should be expected to star in a 7/10 rated film and his ratings tend to have a standard deviation of 0.75.
           Then the "Primary Actor Relative Ratings" for that primary actor is calculated as:
        """
        )
     st.write("""
               PARR = (Film Rating - Actor Expected Rating) / Actor Rating Standard Deviation
    """
    )
     st.write("""
               PARR =  (6 - 7) / 0.75 = -1.33
               """
               )
     st.write("""
           This is the value relative is a single primary actor. If there are multiple primary actors on the film then the average of this is taken as the films "Primary Actor Relative Ratings - Mean"
           These values is taken by each secondary actor and analysed for calculating a secondary actors statisitics:
        - Relative Actor Score - Mean
        - Relative Actor Score - Std
           """
         )
         
st.write("Here it is observed that the actors from the high performing group (red), then to have a smaller standard deviation, suggesting that they are more consistent with their selection of films")
st.write("This could be caused by a number of factors such as actor desirability, actor skill, quality of agent, the type of projects the actor goes for etc")


col1, col2, col3 = st.columns(3)
col1.image(md_secondary_actors_least_20_films_relative_rating_above_below_05_Mean_STD_only, use_column_width=True)
st.text("")
st.text("")
st.text("")

#%% The million euro question

st.title("The Million Dollar Question: Who is the Worst Supporting Actor in Hollywood?")
st.title("")
st.write("""
         Potentially a mean question, obviously this result isn't a reflection of the actor themselves. This just a quirk of the films they have been in but regardless, their acting career is far better then mine!
         """
       )

col1, col2 , col3 = st.columns(3)
col2.image(FG_acting_career , use_column_width=True)

st.write("""
         So good on them for making it, doing something they love!
         """
       )
st.write("""

         Filtering for actors with a film count of above 10, the secondary actor with the lowest relative score is can be found quite quickly by the following code:
          
             >> md_secondary_actors_filtered_for_count = md_secondary_actors[md_secondary_actors["Count"] >= 20]
          
             >> worst_secondary_actor = md_secondary_actors_filtered_for_count.loc[md_secondary_actors_filtered_for_count['Relative Actor Score - Mean'] == md_secondary_actors_filtered_for_count['Relative Actor Score - Mean'].min()]
             
             
            
            And the results is..... 
        """
        )
st.write("")    
st.write("")
st.write("")
st.write("")

col1, col2 , col3 = st.columns(3)
col2.image(rosie_o_donnell , use_column_width=True)
col2.write("Rose O'Donnell")
st.write("Surely undeserved, likely just the target of internet dweebs who spend all their time writing web pages about film rating statistics")
    
            
            
            
              
              
          


#%% #Layout Block - Forecasting Introduction
st.title("Forecasting")



st.header("Data Input Derivation")
st.write("""
         The forecasting model aims to predict the IMDB rating of a film, from an input made of 6 (at the time of writing) expected from actors involved with the film. First every primary actor linked with the film (starting with the actors with the highest “film count”) have a meta-score populated, then followed by any secondary actors (starting with the highest “film count”)
         
         For primary actors the expected score is taken from a linear regression over their film ratings (as pictured earlier), for secondary actors their expected score equals their average film rating
         
         There are always 6 input data points for a film, therefore in the case there are too many actors, the actors (beginning with secondary) with lower film counts are excluded. In the case there are not enough actors, the later data points are populated with a zero

         “Film Count” – the number of films/series (titles) the actor is linked to
      """
    )



st.header("Forecasting Model")
st.write("""
         The forecasting model can be found in the file [programming_analysis_project_analysis_file.py]
        """         
        )

    
    
    


st.text("")
st.text("")
st.text("")





#%% Modelling Discussion
st.title("Modelling Difficulties")

st.write("""         
         - Towards the end of my modelling I attempted to implement a system were the "metadata" (actors statistics based on film ratings), would only be based on films within the training set. Therefore the metascores produced for testing wouldn't be using any information from the testing set films
             - This caused issues as it also filtered out any data relating a secondary actor to a film from the testing set of films. This was an issue for generating testing metadata. 
             - At the time this issue was encountered it was taking around 7-9 hours to regenerate results
             - Therefore the metadata from the training set was re-split and the forecasting model trained and tested from it.
         - I was unable to use the code [>>import functions.py] or the [runcell] functions in the streamlit application
             - Therefore I've had to place all the general functions at the start of the stream_lit_script.py file. Affecting readability
         - It was a lot of work to generate the metadata
             - Circa 2k lines of coding at time of writing
             - Re-generating the data takes around 9 hours
             - Some of this effort could have been avoided by better data-schema planning (not covered yet in course) or a better knowledge of various methods of combining and operating with tables (learning in progress)
        - I’ve applied better organisation to the coding on [programming_analysis_project_analysis_file.py] and [stream_lit_script.py]. I’ve ran out of time to safely clean up the coding on [programming_analysis_project_data_prep.py]
        - This was a good learning experience all-in-all as I've learnt some hard lessons and have a list of items I want to revise
        - Was unable to link my images/datatables to their permalinks. So I've placed relative file locations (see lines 186, 204 for an example in the stream_lit_script.py file)' 
         """
         )
         

