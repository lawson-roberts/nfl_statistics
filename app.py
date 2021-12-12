import streamlit as st
import pandas as pd
import numpy as np
import base64
import csv
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
## web crawling packages
from pandas import json_normalize
import json
import time
from datetime import date
from datetime import datetime
import datetime

import requests
import pickle
import io

##Setting Streamlit Settings
#st.set_page_config(layout="wide")

## loading historical games
#historical = pd.read_csv('data/nfl_historical_clean.csv')
#historical = historical.drop(columns='Unnamed: 0')
#historical['year'] = historical['url'].str[-17:-13]
#historical['year'] = historical['year'].astype(int)
#today = date.today()
#year = today.year
#historical = historical[historical['year'] == year]

historical = pd.read_csv('data/agg_team_stats.csv')
historical = historical.drop(columns='Unnamed: 0')

## declaring model name that I would like to use
filename = 'models/random_forest_model_1.sav'
# load the model from disk
model_obj = pickle.load(open(filename, 'rb'))

## loading predictions file to show accuracy
predictions_hist = pd.read_csv('predictions/model_prediction_file.csv')
predictions_hist = predictions_hist.drop(columns = 'Unnamed: 0')
game_score_hist = pd.read_csv('data/game_scores.csv')
game_score_hist = game_score_hist.drop(columns = 'Unnamed: 0')

predictions_hist = predictions_hist.merge(game_score_hist, on='id')
predictions_hist['tie_check'] = np.where(predictions_hist['away_score'] ==  predictions_hist['home_score'], 'tie', 'no tie')
predictions_hist = predictions_hist[predictions_hist['tie_check'] == 'no tie']
predictions_hist['winner'] = np.where(predictions_hist['away_score'] >  predictions_hist['home_score'], 'away', 'home')
predictions_hist['prediction'] = np.where(predictions_hist['away'] >  predictions_hist['home'], 'away', 'home')
predictions_hist['pred_correct'] = np.where(predictions_hist['winner'] == predictions_hist['prediction'], 1, 0)
predictions_hist['away'] = round(predictions_hist['away']*100, 2)
predictions_hist['home'] = round(predictions_hist['home']*100, 2)

model_score = round((sum(predictions_hist['pred_correct']) / len(predictions_hist))*100, 2)

st.title("NFL Game Predictions")
st.write("""
        ## - Using machine learning to find best odds of winning certain sports betting wagers
        """)
st.write("""### Data Sources:""")
st.write("""1.) https://www.espn.com/nfl/scoreboard/ used for getting game statistics""")
st.write("""2.) Caesars Sportsbook used for vegas bets available.""")

st.write("""### Model Results thus far...""")
st.write("Accuracy Score", model_score, "%")
st.write("""#### Game Detail for past predictions...""")
st.write(predictions_hist.astype('object'))

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

#col1, col2, col3 = st.beta_columns(3)
col1, col2 = st.beta_columns(2)

for ind in game_df.index:

    away = game_df['away'][ind]
    away_logo_url =  "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/" + str(away) + ".png&scale=crop&cquality=40&location=origin&w=64&h=64"
    away_logo_response = requests.get(away_logo_url)
    away_image_bytes = io.BytesIO(away_logo_response.content)
    away_img = Image.open(away_image_bytes)

    home = game_df['home'][ind]
    home_logo_url =  "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/" + str(home) + ".png&scale=crop&cquality=40&location=origin&w=64&h=64"
    home_logo_response = requests.get(home_logo_url)
    home_image_bytes = io.BytesIO(home_logo_response.content)
    home_img = Image.open(home_image_bytes)

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

    prediction = model_obj.predict(stats_all)
    probabilities = pd.DataFrame(model_obj.predict_proba(stats_all), columns=['away', 'home'])
    probabilities = probabilities.rename(columns = {'away': away, 'home': home})
    prediction_value = prediction[0]

    with col1:
        st.image(home_img)
        st.write("Home Team:", game_df['home'][ind])
        st.write("Home Team Money Line:", game_df['odds.home.moneyLine'][ind])
        st.write("Model Predicts...", round(probabilities[home][0]*100, 2), "% chance of winning")
        st.write("-------------------------")

    with col2:
        st.image(away_img)
        st.write("Away Team:", game_df['away'][ind])
        st.write("Away Team Money Line:", game_df['odds.away.moneyLine'][ind])
        st.write("Model Predicts...", round(probabilities[away][0]*100, 2), "% chance of winning")
        #st.write("")
        #t.write("")
        st.write("-------------------------")

    #with col3:
        #st.write(stats_all.astype('object'))

        #team = {'away': away, 'home': home}
        #if prediction_value == "home":
            #st.image(home_img)
            #st.write("Model Predicts...", home, " as the winner.")
        #else:
            #st.image(away_img)
            #st.write("Model Predicts...", away, " as the winner")

        #st.write("Model Probabilities below...", probabilities)
        #st.write("Start Time:", game_df['date'][ind], game_df['time'][ind], " PM CST")
        #st.write("Odds for Spread:", game_df['odds.details'][ind], "---", "Over/Under:", game_df['odds.overUnder'][ind])
        #st.write("Over/Under:", game_df['odds.overUnder'][ind])
        #st.write("Over/Under:", game_df['odds.overUnder'][ind], "Odds", game_df['odds.overOdds'][ind])
        #st.write("-------------------------")

st.write(game_df.astype('object'))
#st.write(game_dict)