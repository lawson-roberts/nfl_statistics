### importing packages
import pandas as pd
import numpy as np
from pandas import json_normalize
import json
#from selenium.webdriver.support.expected_conditions import element_selection_state_to_be
#import matplotlib.pyplot as plt
import base64
#import matplotlib.pyplot as plt
import io
import os
from math import floor

## web crawling packages
import time
from datetime import date
from datetime import datetime
import datetime
import requests
from lxml import html
import csv
import urllib.request
from html_table_parser.parser import HTMLTableParser
from pprint import pprint

### first we need to collect last weeks game id's

#### load in last weeks games

### second we need to scrap those game stats
game_stats_big_df = pd.DataFrame()

for i in game_id_df['id']:
    
    try:
        web = "https://www.espn.com/nfl/matchup?gameId=" + str(i)

        # defining the html contents of a URL.
        xhtml = url_get_contents(web).decode('utf-8')

        # Defining the HTMLTableParser object
        p = HTMLTableParser()

        # feeding the html contents in the
        # HTMLTableParser object
        p.feed(xhtml)

        # Now finally obtaining the data of
        # the table required
        df_testing = pd.DataFrame(p.tables[1])
        column_list = list(df_testing[0].values)

        away = pd.DataFrame(df_testing[1]).T
        away.columns = column_list
        away = away.reset_index()
        away = away.drop(columns = ['index', 'Matchup'])
        #away.drop(away.columns[[13]], axis=1, inplace=True)
        away = away.rename(columns = {'1st Downs': 'first_downs_away',
                            'Passing 1st downs': 'first_downs_passing_away',
                            'Rushing 1st downs': 'first_downs_rushing_away', 
                            '1st downs from penalties': 'first_down_by_penalty_away', 
                            '3rd down efficiency': 'third_down_away', 
                            '4th down efficiency': 'fourth_down_away', 
                            'Total Plays': 'total_plays_away', 
                            'Total Yards': 'total_yards_away', 
                            'Total Drives': 'total_drives_away', 
                            'Yards per Play': 'yards_per_play_away', 
                            'Passing': 'passing_away', 
                            'Comp-Att': 'comp_att_away', 
                            'Yards per pass': 'yards_per_pass_away',
                            'Fumbles Lost': 'fumbles_lost_away',  
                            'Sacks-Yards Lost': 'sack_yards_lost_away', 
                            'Rushing': 'rushing_away', 
                            'Rushing Attempts': 'rushing_att_away', 
                            'Yards per rush': 'yards_per_rush_away', 
                            'Red Zone (Made-Att)': 'red_zone_att_away', 
                            'Penalties': 'penalty_away', 
                            'Turnovers': 'turnovers_away', 
                            'Interceptions thrown': 'int_thrown_away', 
                            'Rushing 1st downs': 'first_down_rushing_away', 
                            'Defensive / Special Teams TDs': 'defensive_td_away', 
                            'Possession': 'possession_away'})

        home = pd.DataFrame(df_testing[2]).T
        home.columns = column_list
        home = home.reset_index()
        home = home.drop(columns = ['index', 'Matchup'])
        #home.drop(home.columns[[13]], axis=1, inplace=True)
        home = home.rename(columns = {'1st Downs': 'first_downs_home',
                            'Passing 1st downs': 'first_downs_passing_home',
                            'Rushing 1st downs': 'first_downs_rushing_home', 
                            '1st downs from penalties': 'first_down_by_penalty_home', 
                            '3rd down efficiency': 'third_down_home', 
                            '4th down efficiency': 'fourth_down_home', 
                            'Total Plays': 'total_plays_home', 
                            'Total Yards': 'total_yards_home', 
                            'Total Drives': 'total_drives_home', 
                            'Yards per Play': 'yards_per_play_home', 
                            'Passing': 'passing_home', 
                            'Comp-Att': 'comp_att_home', 
                            'Yards per pass': 'yards_per_pass_home',
                            'Fumbles Lost': 'fumbles_lost_home',  
                            'Sacks-Yards Lost': 'sack_yards_lost_home', 
                            'Rushing': 'rushing_home', 
                            'Rushing Attempts': 'rushing_att_home', 
                            'Yards per rush': 'yards_per_rush_home', 
                            'Red Zone (Made-Att)': 'red_zone_att_home', 
                            'Penalties': 'penalty_home', 
                            'Turnovers': 'turnovers_home', 
                            'Interceptions thrown': 'int_thrown_home', 
                            'Rushing 1st downs': 'first_down_rushing_home', 
                            'Defensive / Special Teams TDs': 'defensive_td_home', 
                            'Possession': 'possession_home'})

        game_stat_temp = pd.concat([away, home], axis=1)
        game_stat_temp['id'] = i
        game_stats_big_df = game_stats_big_df.append(game_stat_temp)

    except Exception as e:
        print("Error:", e)

game_score_big_df = pd.DataFrame()

for i in game_id_df['id']:
    
    try:
        # defining the html contents of a URL.
        web = "https://www.espn.com/nfl/matchup?gameId=" + str(i)

        xhtml = url_get_contents(web).decode('utf-8')

        # Defining the HTMLTableParser object
        p = HTMLTableParser()

        # feeding the html contents in the
        # HTMLTableParser object
        p.feed(xhtml)

        df_score = pd.DataFrame(p.tables[0])
        df_score = pd.DataFrame(df_score.iloc[1: , :])
        team_column = df_score[0].to_frame()
        #team_column = team_column.rename(columns = {0: 'team'})
        #team_df = pd.DataFrame(team_column, columns = ['team'])
        score_column = df_score.iloc[: , -1].to_frame()
        #score_column = score_column.rename({: 'score'})
        #score_df = pd.DataFrame(score_column, columns = ['score'])
        df_score_new = pd.concat([team_column, score_column], axis=1)
        df_score_new.columns = ['team', 'score']

        away_score = pd.DataFrame(df_score_new.iloc[0]).T
        away_score = away_score.reset_index()
        away_score = away_score.drop(columns = ['index'])
        #away_score = away_score[['team', 'score']]
        away_score = away_score.rename(columns = {'team': 'away_team', 'score': 'away_score'})

        home_score = pd.DataFrame(df_score_new.iloc[1]).T
        home_score = home_score.reset_index()
        home_score = home_score.drop(columns = ['index'])
        #home_score = home_score[['team', 'score']]
        home_score = home_score.rename(columns = {'team': 'home_team', 'score': 'home_score'})

        game_score_temp = pd.concat([away_score, home_score], axis=1)
        game_score_temp['id'] = i
        #game_score_temp = game_score_temp.reset_index()
        #game_score_temp = game_score_temp.drop(columns = ['index'])
        game_score_big_df = game_score_big_df.append(game_score_temp)

    except Exception as e:
        print('Error:', e, 'game_id=', i)

def convert_fraction(column_name, new_column_name, df):
    # new data frame with split value columns
    new = df[column_name].str.split("-", n = 1, expand = True)
  
    # making separate first name column from new data frame
    df["numerator"]= new[0]
    df["numerator"] = df["numerator"].astype(int)
  
    # making separate last name column from new data frame
    df["denominator"]= new[1]
    df["denominator"] = df["denominator"].astype(int)

    df[new_column_name] = df["numerator"]/df["denominator"]

    # Dropping old Name columns
    df.drop(columns =[column_name, 'numerator', 'denominator'], inplace = True)
    return df

def convert_att_yards(column_name, df):
    # new data frame with split value columns
    new = df[column_name].str.split("-", n = 1, expand = True)
  
    # making separate first name column from new data frame
    att_col_str = column_name + "_occur"
    yards_col_str = column_name + "_yards"

    df[att_col_str]= new[0]
    df[att_col_str] = df[att_col_str].astype(int)
  
    # making separate last name column from new data frame
    df[yards_col_str]= new[1]
    df[yards_col_str] = df[yards_col_str].astype(int)

    # Dropping old Name columns
    df.drop(columns =[column_name], inplace = True)
    return df

def cleaning_last_weeks_stats(df, game_urls):
    df = convert_fraction("third_down_away", "third_down_away", df)
    df = convert_fraction("third_down_home", "third_down_home", df)
    df = convert_fraction("fourth_down_away", "fourth_down_away", df)
    df = convert_fraction("fourth_down_home", "fourth_down_home", df)
    df = convert_fraction("comp_att_away", "comp_att_away", df)
    df = convert_fraction("comp_att_home", "comp_att_home", df)
    df = df.fillna(0)

    ## need to clean a few other variables that are being treated as str's
    df = convert_att_yards("sack_yards_lost_away", df)
    df = convert_att_yards("sack_yards_lost_home", df)
    df = convert_att_yards("penalty_away", df)
    df = convert_att_yards("penalty_home", df)

    ## change possession into float
    df['possession_away'] = df['possession_away'].str.replace(':', '.')
    df['possession_home'] = df['possession_home'].str.replace(':', '.')
    df['possession_away'] = df['possession_away'].astype(float)
    df['possession_home'] = df['possession_home'].astype(float)

    ## needing to declare the winning team to create target variable
    df['target'] = np.where(df['away_score'] > df['home_score'], 1, 0)
    df['winning_team'] = np.where(df['away_score'] > df['home_score'], df['away_team'], df['home_team'])

    ## merge with game urls file in order to know when games occured
    df = df.merge(game_urls, on='id', how='inner')

### clean stats and append to master historical file

#### this is where we will take these cleaned results and write files back to github
##### files will include cleaned master file, and file that is ready for model training