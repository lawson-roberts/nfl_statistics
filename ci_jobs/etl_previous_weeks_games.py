import pandas as pd
import numpy as np
from math import sqrt
import pickle
import requests
from pandas import json_normalize
import json

#from io import BytesIO
#from io import StringIO
#mport boto3

import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import date
from datetime import datetime
import datetime

import urllib.request
from html_table_parser.parser import HTMLTableParser
from pprint import pprint

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import shap

from statsmodels.stats.outliers_influence import variance_inflation_factor

print("1. updating previous_weeks_games_list")
## pulling previous weeks games
previous_weeks_games = pd.read_csv('predictions/previous_week_games.csv')
# previous_weeks_games = previous_weeks_games.drop(columns = ['Unnamed: 0'])

previous_weeks_games_list = previous_weeks_games['id']


print("1a. scrape game stats for previous_weeks_games_list")
# Opens a website and read its
# binary contents (HTTP Response Body)
def url_get_contents(url):

    # Opens a website and read its
    # binary contents (HTTP Response Body)

    #making request to the website
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)

    #reading contents of the website
    return f.read()

game_stats_big_df = pd.DataFrame()

for i in previous_weeks_games_list:
    
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
        away = away.reset_index(drop=True)
        # away = away.drop(columns = ['index'])
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
                            'Fumbles lost': 'fumbles_lost_away',  
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
        home = home.reset_index(drop=True)
        # home = home.drop(columns = ['index'])
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
                            'Fumbles lost': 'fumbles_lost_home',  
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
        # game_stats_big_df = game_stats_big_df.append(game_stat_temp)
        game_stats_big_df = pd.concat([game_stats_big_df, game_stat_temp], axis=0).reset_index(drop=True)

    except Exception as e:
        print("Error:", e)

try:
    game_stats_big_df = game_stats_big_df.drop(columns = [''])
except:
    pass


print("1b. scrape game results for previous_weeks_games_list")
game_score_big_df = pd.DataFrame()

for i in previous_weeks_games_list:
    
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
        away_score = away_score.reset_index(drop=True)
        # away_score = away_score.drop(columns = ['index'])
        #away_score = away_score[['team', 'score']]
        away_score = away_score.rename(columns = {'team': 'away_team', 'score': 'away_score'})

        home_score = pd.DataFrame(df_score_new.iloc[1]).T
        home_score = home_score.reset_index(drop=True)
        # home_score = home_score.drop(columns = ['index'])
        #home_score = home_score[['team', 'score']]
        home_score = home_score.rename(columns = {'team': 'home_team', 'score': 'home_score'})

        game_score_temp = pd.concat([away_score, home_score], axis=1)
        game_score_temp['id'] = i
        #game_score_temp = game_score_temp.reset_index()
        #game_score_temp = game_score_temp.drop(columns = ['index'])
        # game_score_big_df = game_score_big_df.append(game_score_temp)
        game_score_big_df = pd.concat([game_score_big_df, game_score_temp], axis=0).reset_index(drop=True)

    except Exception as e:
        print('Error:', e, 'game_id=', i)


print("1c. join these three previous_weeks_games, game_stats_big_df, game_score_big_df")
df_ready_to_clean = game_stats_big_df.merge(game_score_big_df, on='id', how='inner')
df_ready_to_clean = df_ready_to_clean.merge(previous_weeks_games, on='id', how='inner')


print("2. Begin Cleaning Stats data")
def convert_fraction(df, column_name, new_column_name):
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

def convert_att_yards(df, column_name):
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

df_ready_to_clean = convert_fraction(df_ready_to_clean, "third_down_away", "third_down_away")
df_ready_to_clean = convert_fraction(df_ready_to_clean, "third_down_home", "third_down_home")
df_ready_to_clean = convert_fraction(df_ready_to_clean, "fourth_down_away", "fourth_down_away")
df_ready_to_clean = convert_fraction(df_ready_to_clean, "fourth_down_home", "fourth_down_home")
df_ready_to_clean = convert_fraction(df_ready_to_clean, "comp_att_away", "comp_att_away")
df_ready_to_clean = convert_fraction(df_ready_to_clean, "comp_att_home", "comp_att_home")
df_ready_to_clean = df_ready_to_clean.fillna(0)

## need to clean a few other variables that are being treated as str's
df_ready_to_clean = convert_att_yards(df_ready_to_clean, "sack_yards_lost_away")
df_ready_to_clean = convert_att_yards(df_ready_to_clean, "sack_yards_lost_home")
df_ready_to_clean = convert_att_yards(df_ready_to_clean, "penalty_away")
df_ready_to_clean = convert_att_yards(df_ready_to_clean, "penalty_home")

## change possession into float
df_ready_to_clean['possession_away'] = df_ready_to_clean['possession_away'].str.replace(':', '.')
df_ready_to_clean['possession_home'] = df_ready_to_clean['possession_home'].str.replace(':', '.')
df_ready_to_clean['possession_away'] = df_ready_to_clean['possession_away'].astype(float)
df_ready_to_clean['possession_home'] = df_ready_to_clean['possession_home'].astype(float)

## needing to declare the winning team to create target variable
df_ready_to_clean['target'] = np.where(df_ready_to_clean['away_score'] > df_ready_to_clean['home_score'], 1, 0)
df_ready_to_clean['winning_team'] = np.where(df_ready_to_clean['away_score'] > df_ready_to_clean['home_score'], df_ready_to_clean['away_team'], df_ready_to_clean['home_team'])


print("2a. creating and updating historical stats file")
## dropping a few last un-needed cloumns
df_ready_to_clean = df_ready_to_clean.loc[:,~df_ready_to_clean.columns.duplicated()]
df_clean = df_ready_to_clean.drop(columns=['red_zone_att_away', 'red_zone_att_home'])

try:
    df_clean = df_clean.drop(columns=[''])
except:
    pass

df_clean_hist = pd.read_csv('data/nfl_historical_clean.csv')
print("Before:", df_clean_hist.shape)
# df_clean_hist = df_clean_hist.drop(columns = 'Unnamed: 0')
# df_clean_hist = df_clean_hist.append(df_clean, ignore_index=True)
df_clean_hist = pd.concat([df_clean_hist, df_clean], axis=0).reset_index(drop=True)
# df_clean_hist = df_clean_hist.drop_duplicates(keep='last').reset_index(drop=True)
print("After:", df_clean_hist.shape)

df_clean_hist.to_csv('data/nfl_historical_clean.csv', index=False)
print("Before: ", df_clean_hist.shape)

print("2b. creating and updating df_ready_for_model.csv")
df_ready_for_model = df_ready_to_clean.drop(columns=['red_zone_att_away', 'red_zone_att_home', 'url', 'winning_team', 'away_team', 'away_score', 'home_team', 'home_score', 'id'])

df_ready_for_model_hist = pd.read_csv('data/df_ready_for_model.csv')
# df_ready_for_model_hist = df_ready_for_model_hist.drop(columns = 'Unnamed: 0')
# df_ready_for_model_hist = df_ready_for_model_hist.append(df_ready_for_model, ignore_index=True)
df_ready_for_model_hist = pd.concat([df_ready_for_model_hist, df_ready_for_model], axis=0).reset_index(drop=True)
print("After: ", df_ready_for_model_hist.shape)
df_ready_for_model_hist.to_csv('data/df_ready_for_model.csv', index=False)
print("Done!")

print("3. Retraining Model")
df_ready_for_model_hist = pd.read_csv('data/df_ready_for_model.csv')
# df_ready_for_model_hist = df_ready_for_model_hist.drop(columns = 'Unnamed: 0')
df_model = df_ready_for_model_hist
df_model['target_class']  = np.where(df_model['target'] == 1, "away", "home")
print(df_model.groupby('target_class').size())
df_model = df_model.drop(columns = ['target'])
# df_model = df_model.drop_duplicates()

data_bucket = df_model.copy()
data_bucket = data_bucket.dropna()
data_len = len(data_bucket.columns)

y_adj_bucket = data_bucket.iloc[:,data_len-1:]
X_adj_bucket = data_bucket.iloc[:,:data_len-1]

rf_clf =RandomForestClassifier(max_depth = 10,n_estimators=40, max_features = 'log2', random_state=42)
#rf_clf =RandomForestClassifier(max_depth = 6,n_estimators=10, max_features = 'log2', random_state=0, class_weight={'0-4%': 1, '4-7.6%': 2})
np.random.seed(42)

##smote
sm = SMOTE(random_state=42)

## training
#X_train_res, y_train_res = sm.fit_resample(x_train, y_train)

## productionize
X_res, y_res = sm.fit_resample(X_adj_bucket, y_adj_bucket)

##random oversample
#ros = RandomOverSampler(random_state=42)
#X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

classifier = rf_clf.fit(X_res, y_res)
#classifier = rf_clf.fit(x_train, y_train)
#print("Model Score is:", rf_clf.score(x_test, y_test))
#y_pred_train = rf_clf.predict(X_train_res)
y_pred = rf_clf.predict(X_res)
#y_pred_test = rf_clf.predict(x_test)

## saving model
# save the model to disk
filename = 'models/random_forest_model_1.sav'
pickle.dump(classifier, open(filename, 'wb'))

# compute SHAP values
explainer = shap.TreeExplainer(rf_clf, X_adj_bucket, check_additivity=False)
shap_values = explainer.shap_values(X_adj_bucket, check_additivity=False)

shap_values_class_0 = pd.DataFrame(shap_values[0])
shap_values_class_1 = pd.DataFrame(shap_values[1])

shap_values_class_0.to_csv('shap_values/shap_values_class_0.csv')
shap_values_class_1.to_csv('shap_values/shap_values_class_1.csv')

print("4. Prepare teams historical stats file and update model")
stats_agg_df = pd.read_csv('data/nfl_historical_clean.csv')
# stats_agg_df = stats_agg_df.drop(columns = 'Unnamed: 0')
# stats_agg_df = stats_agg_df.drop_duplicates()

stats_agg_df['year'] = stats_agg_df['url'].str[-17:-13]
stats_agg_df['year'] = stats_agg_df['year'].astype(int)
today = date.today()
year = today.year
stats_agg_df = stats_agg_df[(stats_agg_df['year'] == year) | (stats_agg_df['year'] == year-1)]

print("4. Prepare teams historical stats file")
stats_agg_df = pd.read_csv('data/nfl_historical_clean.csv')
# stats_agg_df = stats_agg_df.drop(columns = 'Unnamed: 0')
# stats_agg_df = stats_agg_df.drop_duplicates()

stats_agg_df['year'] = stats_agg_df['url'].str[-17:-13]
stats_agg_df['year'] = stats_agg_df['year'].astype(int)
today = date.today()
year = today.year
stats_agg_df = stats_agg_df[(stats_agg_df['year'] == year) | (stats_agg_df['year'] == year-1)]

team_list = stats_agg_df['away_team'].unique()

stat_agg_df = pd.DataFrame()

for i in team_list:
    team = i

    new_DataFrame = stats_agg_df[(stats_agg_df['away_team'] == team) | (stats_agg_df['home_team'] == team)]

    stat_df_away = new_DataFrame[new_DataFrame['away_team'] == team]
    stat_df_away = stat_df_away[['first_downs_away', 'first_downs_passing_away', 'first_down_rushing_away', 'first_down_by_penalty_away', 'total_plays_away', 'total_yards_away', 'total_drives_away', 'yards_per_play_away', 'passing_away', 'yards_per_pass_away', 'int_thrown_away', 'rushing_away', 'rushing_att_away', 'yards_per_rush_away', 'turnovers_away', 'fumbles_lost_away', 'defensive_td_away', 'possession_away', 'sack_yards_lost_away_occur', 'sack_yards_lost_away_yards', 'penalty_away_occur', 'penalty_away_yards']]
    stat_df_away = stat_df_away.rename(columns = {'first_downs_away': 'first_downs', 'first_downs_passing_away': 'first_downs_passing', 'first_down_rushing_away': 'first_down_rushing', 'first_down_by_penalty_away': 'first_down_by_penalty', 'total_plays_away': 'total_plays', 'total_yards_away': 'total_yards', 'total_drives_away': 'total_drives', 'yards_per_play_away': 'yards_per_play', 'passing_away': 'passing', 'yards_per_pass_away': 'yards_per_pass', 'int_thrown_away': 'int_thrown', 'rushing_away': 'rushing', 'rushing_att_away': 'rushing_att', 'yards_per_rush_away': 'yards_per_rush', 'turnovers_away': 'turnovers', 'fumbles_lost_away': 'fumbles_lost', 'defensive_td_away': 'defensive_td', 'possession_away': 'possession', 'sack_yards_lost_away_occur': 'sack_yards_lost_occur', 'sack_yards_lost_away_yards': 'sack_yards_lost_yards', 'penalty_away_occur': 'penalty_occur', 'penalty_away_yards': 'penalty_yards'})
    
    stat_df_home = new_DataFrame[new_DataFrame['home_team'] == team]
    stat_df_home = stat_df_home[['first_downs_home', 'first_downs_passing_home', 'first_down_rushing_home', 'first_down_by_penalty_home', 'total_plays_home', 'total_yards_home', 'total_drives_home', 'yards_per_play_home', 'passing_home', 'yards_per_pass_home', 'int_thrown_home', 'rushing_home', 'rushing_att_home', 'yards_per_rush_home', 'turnovers_home', 'fumbles_lost_home', 'defensive_td_home', 'possession_home', 'sack_yards_lost_home_occur', 'sack_yards_lost_home_yards', 'penalty_home_occur', 'penalty_home_yards']]
    stat_df_home = stat_df_home.rename(columns = {'first_downs_home': 'first_downs', 'first_downs_passing_home': 'first_downs_passing', 'first_down_rushing_home': 'first_down_rushing', 'first_down_by_penalty_home': 'first_down_by_penalty', 'total_plays_home': 'total_plays', 'total_yards_home': 'total_yards', 'total_drives_home': 'total_drives', 'yards_per_play_home': 'yards_per_play', 'passing_home': 'passing', 'yards_per_pass_home': 'yards_per_pass', 'int_thrown_home': 'int_thrown', 'rushing_home': 'rushing', 'rushing_att_home': 'rushing_att', 'yards_per_rush_home': 'yards_per_rush', 'turnovers_home': 'turnovers', 'fumbles_lost_home': 'fumbles_lost', 'defensive_td_home': 'defensive_td', 'possession_home': 'possession', 'sack_yards_lost_home_occur': 'sack_yards_lost_occur', 'sack_yards_lost_home_yards': 'sack_yards_lost_yards', 'penalty_home_occur': 'penalty_occur', 'penalty_home_yards': 'penalty_yards'})

    # team_stats_all = stat_df_away.append(stat_df_home)
    team_stats_all = pd.concat([stat_df_away, stat_df_home], axis=0).reset_index(drop=True)
    team_stats_agg_temp = pd.DataFrame(team_stats_all.mean()).T
    team_stats_agg_temp['team'] = team
    # stat_agg_df = stat_agg_df.append(team_stats_agg_temp)
    stat_agg_df = pd.concat([stat_agg_df, team_stats_agg_temp], axis=0).reset_index(drop=True)
    print(i, "Done")

print("All Done")

# stat_agg_df = stat_agg_df.drop_duplicates()
stat_agg_df.to_csv('data/agg_team_stats.csv', index=False)

print("5. Update game historical files")
game_id_og = pd.read_csv('data/game_id_all.csv')
# game_id_og = game_id_og.drop(columns = ['Unnamed: 0'])
previous_weeks_games_df = previous_weeks_games
# game_id_all = game_id_og.append(previous_weeks_games_df)
game_id_all = pd.concat([game_id_og, previous_weeks_games_df], axis=0).reset_index(drop=True)
# game_id_all = game_id_all.drop_duplicates()
game_id_all = game_id_all.dropna()

game_scores_og = pd.read_csv('data/game_scores.csv')
# game_scores_og = game_scores_og.drop(columns = ['Unnamed: 0'])
# game_scores_all = game_scores_og.append(game_score_big_df)
game_scores_all = pd.concat([game_scores_og, game_score_big_df], axis=0).reset_index(drop=True)
# game_scores_all = game_scores_all.drop_duplicates()
game_scores_all.to_csv('data/game_scores.csv', index=False)
print("Done!")

print("6. create previous weeks games file")
historical = pd.read_csv('data/agg_team_stats.csv')
probs_historical = pd.read_csv('predictions/model_prediction_file.csv')
# probs_historical = pd.DataFrame(columns = ['away', 'home', 'id'])
# historical = historical.drop(columns='Unnamed: 0')
# probs_historical = probs_historical.drop(columns='Unnamed: 0')

## declaring model name that I would like to use
filename = 'models/random_forest_model_1.sav'
# load the model from disk
model_obj = pickle.load(open(filename, 'rb'))

def get_current_games():
    url = "https://site.web.api.espn.com/apis/v2/scoreboard/header?sport=football&league=nfl&region=us&lang=en&contentorigin=espn&buyWindow=1m&showAirings=buy%2Clive%2Creplay&showZipLookup=true&tz=America/New_York"

    payload={}
    headers = {
    'Accept': 'application/json',
    'Cookie': 'SWID=3750444F-BA06-4929-C99C-62333D7EE8A0'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    response_text = json.loads(response.text)

    game_df = json_normalize(response_text['sports'], ['leagues', 'events'])
    game_dict = game_df.to_dict('index')
    return game_df, game_dict

game_df, game_dict = get_current_games()
game_df[['away','home']] = game_df.shortName.apply(lambda x: pd.Series(str(x).split("@")))
game_df['away'] = game_df['away'].str.strip()
game_df['home'] = game_df['home'].str.strip()
game_df[['date','time']] = game_df.date.apply(lambda x: pd.Series(str(x).split("T")))
game_df['time'] = game_df['time'].str[:5]
game_df['time'] = game_df['time'].str.replace("18", "12")
game_df['time'] = game_df['time'].str.replace("21", "03")
game_df['time'] = game_df['time'].str.replace("01", "07")

previous_week_games_file_df = game_df[['id', 'season', 'seasonType', 'week']]
previous_week_games_file_df['season'] = previous_week_games_file_df['season'].astype(str)
previous_week_games_file_df['week'] = previous_week_games_file_df['week'].astype(str)
previous_week_games_file_df['url'] = "https://www.espn.com/nfl/scoreboard/_/week/" + previous_week_games_file_df['week'] + "/year/" + previous_week_games_file_df['season'] + "/seasontype/" + previous_week_games_file_df['seasonType']
previous_week_games_file_df = previous_week_games_file_df.drop(columns = ['season', 'seasonType', 'week'])
previous_week_games_file_df.to_csv('predictions/previous_week_games.csv', index=False)

probabilities_df = pd.DataFrame()
column_order = X_res.columns

for ind in game_df.index:

    game_id = game_df['id'][ind]
    #id_list.append(game_id)

    away = game_df['away'][ind]
    home = game_df['home'][ind]

    away_stat = game_df[game_df['away'] == away]
    away_stat = pd.DataFrame(away_stat['away'])
    away_stat = away_stat.rename(columns = {'away': 'team'})
    #away_stat.columns = ['team']
    away_stat = away_stat.merge(historical, on='team', how='inner')
    away_stat = away_stat.rename(columns = {'first_downs':'first_downs_away', 'first_downs_passing':'first_downs_passing_away', 'first_down_rushing':'first_down_rushing_away', 'first_down_by_penalty':'first_down_by_penalty_away', 'total_plays':'total_plays_away', 'total_yards':'total_yards_away', 'total_drives':'total_drives_away', 'yards_per_play':'yards_per_play_away', 'passing':'passing_away', 'yards_per_pass':'yards_per_pass_away', 'int_thrown':'int_thrown_away', 'rushing':'rushing_away', 'rushing_att':'rushing_att_away', 'yards_per_rush':'yards_per_rush_away', 'turnovers':'turnovers_away', 'fumbles_lost':'fumbles_lost_away', 'defensive_td':'defensive_td_away', 'possession':'possession_away', 'sack_yards_lost_occur':'sack_yards_lost_away_occur', 'sack_yards_lost_yards':'sack_yards_lost_away_yards', 'penalty_occur':'penalty_away_occur', 'penalty_yards':'penalty_away_yards'})
    away_stat = away_stat.reset_index()
    away_stat = away_stat.drop(columns = ['index', 'team'])

    home_stat = game_df[game_df['home'] == home]
    home_stat = pd.DataFrame(home_stat['home'])
    home_stat = home_stat.rename(columns = {'home': 'team'})
    #home_stat.columns = ['team']
    home_stat = home_stat.merge(historical, on='team', how='inner')
    home_stat = home_stat.rename(columns = {'first_downs':'first_downs_home', 'first_downs_passing':'first_downs_passing_home', 'first_down_rushing':'first_down_rushing_home', 'first_down_by_penalty':'first_down_by_penalty_home', 'total_plays':'total_plays_home', 'total_yards':'total_yards_home', 'total_drives':'total_drives_home', 'yards_per_play':'yards_per_play_home', 'passing':'passing_home', 'yards_per_pass':'yards_per_pass_home', 'int_thrown':'int_thrown_home', 'rushing':'rushing_home', 'rushing_att':'rushing_att_home', 'yards_per_rush':'yards_per_rush_home', 'turnovers':'turnovers_home', 'fumbles_lost':'fumbles_lost_home', 'defensive_td':'defensive_td_home', 'possession':'possession_home', 'sack_yards_lost_occur':'sack_yards_lost_home_occur', 'sack_yards_lost_yards':'sack_yards_lost_home_yards', 'penalty_occur':'penalty_home_occur', 'penalty_yards':'penalty_home_yards'})
    home_stat = home_stat.reset_index()
    home_stat = home_stat.drop(columns = ['index', 'team'])  

    stats_all = pd.concat([away_stat, home_stat], axis=1)
    
    # order columns to match model
    stats_all = stats_all[column_order]

    prediction_temp = model_obj.predict(stats_all)
    #prediction_value = str(prediction_temp[0])
    #predictions_list = predictions_list.append([prediction_value])
    probabilities_temp = pd.DataFrame(model_obj.predict_proba(stats_all), columns=['away', 'home'])
    probabilities_temp['id'] = game_id
    #probabilities_temp = probabilities_temp.rename(columns = {'away': away, 'home': home})
    # probabilities_df = probabilities_df.append(probabilities_temp)
    probabilities_df = pd.concat([probabilities_df, probabilities_temp], axis=0).reset_index(drop=True)

probs_historical = pd.concat([probs_historical, probabilities_df], axis=0).reset_index(drop=True)
probs_historical.to_csv('predictions/model_prediction_file.csv', index=False)
print("Done with everything. Ready to commit.")