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
import matplotlib.pyplot as plt
import shap
from shap import Explanation
shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)

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
# historical = historical.drop(columns='Unnamed: 0')

## declaring model name that I would like to use
filename = 'models/random_forest_model_1.sav'
# load the model from disk
model_obj = pickle.load(open(filename, 'rb'))

# df_ready_for_model_hist = pd.read_csv('data/df_ready_for_model.csv')
# # df_ready_for_model_hist = df_ready_for_model_hist.drop(columns = 'Unnamed: 0')
# df_model = df_ready_for_model_hist
# df_model['target_class']  = np.where(df_model['target'] == 1, "away", "home")
# print(df_model.groupby('target_class').size())
# df_model = df_model.drop(columns = ['target'])
# df_model = df_model.drop_duplicates()

# data_bucket = df_model.copy()
# data_bucket = data_bucket.dropna()
# data_len = len(data_bucket.columns)

# y_adj_bucket = data_bucket.iloc[:,data_len-1:]
# X_adj_bucket = data_bucket.iloc[:,:data_len-1]



## loading predictions file to show accuracy
predictions_hist = pd.read_csv('predictions/model_prediction_file.csv')
# predictions_hist = predictions_hist.drop(columns = 'Unnamed: 0')
game_score_hist = pd.read_csv('data/game_scores.csv')
# game_score_hist = game_score_hist.drop(columns = 'Unnamed: 0')

# predictions_hist = predictions_hist.merge(game_score_hist, on='id')
# predictions_hist['tie_check'] = np.where(predictions_hist['away_score'] ==  predictions_hist['home_score'], 'tie', 'no tie')
# predictions_hist = predictions_hist[predictions_hist['tie_check'] == 'no tie']
# predictions_hist['winner'] = np.where(predictions_hist['away_score'] >  predictions_hist['home_score'], 'away', 'home')
# predictions_hist['prediction'] = np.where(predictions_hist['away'] >  predictions_hist['home'], 'away', 'home')
# predictions_hist['pred_correct'] = np.where(predictions_hist['winner'] == predictions_hist['prediction'], 1, 0)
# predictions_hist['away'] = round(predictions_hist['away']*100, 2)
# predictions_hist['home'] = round(predictions_hist['home']*100, 2)

# model_score = round((sum(predictions_hist['pred_correct']) / len(predictions_hist))*100, 2)

st.title("NFL Game Predictions")
st.write("""
        ## - Using machine learning to find best odds of winning certain sports betting wagers
        """)
st.write("""### Data Sources:""")
st.write("""1.) https://www.espn.com/nfl/scoreboard/ used for getting game statistics""")
st.write("""2.) Caesars Sportsbook used for vegas bets available.""")

# st.write("""### Model Results thus far...""")
# st.write("Accuracy Score", model_score, "%")
# st.write("""#### Game Detail for past predictions...""")
# st.write(predictions_hist.astype('object'))

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

away_df = game_df[['away']].copy(deep=True)
away_df = away_df.rename(columns = {'away': 'team'})
away_df = away_df.merge(historical, on='team', how='inner')
away_df = away_df.rename(columns = {'first_downs':'first_downs_away', 'first_downs_passing':'first_downs_passing_away', 'first_down_rushing':'first_down_rushing_away', 'first_down_by_penalty':'first_down_by_penalty_away', 'total_plays':'total_plays_away', 'total_yards':'total_yards_away', 'total_drives':'total_drives_away', 'yards_per_play':'yards_per_play_away', 'passing':'passing_away', 'yards_per_pass':'yards_per_pass_away', 'int_thrown':'int_thrown_away', 'rushing':'rushing_away', 'rushing_att':'rushing_att_away', 'yards_per_rush':'yards_per_rush_away', 'turnovers':'turnovers_away', 'fumbles_lost':'fumbles_lost_away', 'defensive_td':'defensive_td_away', 'possession':'possession_away', 'sack_yards_lost_occur':'sack_yards_lost_away_occur', 'sack_yards_lost_yards':'sack_yards_lost_away_yards', 'penalty_occur':'penalty_away_occur', 'penalty_yards':'penalty_away_yards'})
away_df = away_df.reset_index(drop=True)
away_df = away_df.drop(columns = ['team'])

home_df = game_df[['home']].copy(deep=True)
home_df = home_df.rename(columns = {'home': 'team'})
home_df = home_df.merge(historical, on='team', how='inner')
home_df = home_df.rename(columns = {'first_downs':'first_downs_home', 'first_downs_passing':'first_downs_passing_home', 'first_down_rushing':'first_down_rushing_home', 'first_down_by_penalty':'first_down_by_penalty_home', 'total_plays':'total_plays_home', 'total_yards':'total_yards_home', 'total_drives':'total_drives_home', 'yards_per_play':'yards_per_play_home', 'passing':'passing_home', 'yards_per_pass':'yards_per_pass_home', 'int_thrown':'int_thrown_home', 'rushing':'rushing_home', 'rushing_att':'rushing_att_home', 'yards_per_rush':'yards_per_rush_home', 'turnovers':'turnovers_home', 'fumbles_lost':'fumbles_lost_home', 'defensive_td':'defensive_td_home', 'possession':'possession_home', 'sack_yards_lost_occur':'sack_yards_lost_home_occur', 'sack_yards_lost_yards':'sack_yards_lost_home_yards', 'penalty_occur':'penalty_home_occur', 'penalty_yards':'penalty_home_yards'})
home_df = home_df.reset_index(drop=True)
home_df = home_df.drop(columns = ['team'])

stats_all = pd.concat([away_df, home_df], axis=1)
column_order = ['first_downs_away', 'first_downs_passing_away', 'first_down_rushing_away', 'first_down_by_penalty_away', 'total_plays_away', 'total_yards_away', 'total_drives_away', 'yards_per_play_away', 'passing_away', 'yards_per_pass_away', 'int_thrown_away', 'rushing_away', 'rushing_att_away', 'yards_per_rush_away', 'turnovers_away', 'fumbles_lost_away', 'defensive_td_away', 'possession_away', 'first_downs_home', 'first_downs_passing_home', 'first_down_rushing_home', 'first_down_by_penalty_home', 'total_plays_home', 'total_yards_home', 'total_drives_home', 'yards_per_play_home', 'passing_home', 'yards_per_pass_home', 'int_thrown_home', 'rushing_home', 'rushing_att_home', 'yards_per_rush_home', 'turnovers_home', 'fumbles_lost_home', 'defensive_td_home', 'possession_home', 'sack_yards_lost_away_occur', 'sack_yards_lost_away_yards', 'sack_yards_lost_home_occur', 'sack_yards_lost_home_yards', 'penalty_away_occur', 'penalty_away_yards', 'penalty_home_occur', 'penalty_home_yards']

# order columns to match model
stats_all = stats_all[column_order]

predictions = pd.DataFrame(model_obj.predict(stats_all), columns=['prediction'])
probabilities = pd.DataFrame(model_obj.predict_proba(stats_all), columns=['away_prob', 'home_prob'])

game_df_reduced = game_df[['name' ,'odds.details', 'odds.home.moneyLine', 'odds.away.moneyLine', 'odds.overUnder', 'odds.overOdds'
                           , 'odds.underOdds', 'odds.spread', 'odds.awayTeamOdds.spreadOdds', 'odds.homeTeamOdds.spreadOdds'
                           , 'id', 'date', 'away', 'home', 'time']].copy(deep=True)

# concat predictions and probabilities to game_df
game_df_pred = pd.concat([game_df_reduced, predictions, probabilities], axis=1)

# st.write(game_df_pred)
bet_amount = st.number_input('Enter your bet amount:', min_value=5, max_value=100000, value=100, step=1, format=None, key=None)
st.write("Bet Amount: $", bet_amount)

# calculate money line odds payout based on $100 bet
game_df_pred['away_payout_moneyline'] = np.where(game_df_pred['odds.away.moneyLine'] < 0, (bet_amount / (abs(game_df_pred['odds.away.moneyLine'])/100)) + bet_amount, bet_amount * (1 + (game_df_pred['odds.away.moneyLine']/100)))
game_df_pred['away_payout_prob_adjusted'] = game_df_pred['away_payout_moneyline'] * game_df_pred['away_prob']

game_df_pred['home_payout_moneyline'] = np.where(game_df_pred['odds.home.moneyLine'] < 0, (bet_amount / (abs(game_df_pred['odds.home.moneyLine'])/100)) + bet_amount, bet_amount * (1 + (game_df_pred['odds.home.moneyLine']/100)))
game_df_pred['home_payout_prob_adjusted'] = game_df_pred['home_payout_moneyline'] * game_df_pred['home_prob']

# create column that holds the best payout
game_df_pred['best_payout'] = np.where(game_df_pred['away_payout_prob_adjusted'] > game_df_pred['home_payout_prob_adjusted'], game_df_pred['away_payout_prob_adjusted'], game_df_pred['home_payout_prob_adjusted'])

# create column that ranks best payout
game_df_pred['best_payout_rank'] = game_df_pred['best_payout'].rank(ascending=False)

# sort game_df_pred by best payout
game_df_pred = game_df_pred.sort_values(by=['best_payout'], ascending=False)

#sort game_df_pred back by index
# game_df_pred = game_df_pred.sort_index()
# st.write(game_df_pred)

# Create an explainer
explainer = shap.Explainer(model_obj)
# Get feature importance values
shap_values = explainer.shap_values(stats_all)
shap_df = pd.DataFrame(shap_values[0])

# col1, col2, col3 = st.columns(3)
col1, col2 = st.columns(2)

for ind in game_df_pred.index:

    away = game_df_pred['away'][ind]
    away_logo_url =  "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/" + str(away) + ".png&scale=crop&cquality=40&location=origin&w=64&h=64"
    away_logo_response = requests.get(away_logo_url)
    away_image_bytes = io.BytesIO(away_logo_response.content)
    away_img = Image.open(away_image_bytes)

    home = game_df_pred['home'][ind]
    home_logo_url =  "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/" + str(home) + ".png&scale=crop&cquality=40&location=origin&w=64&h=64"
    home_logo_response = requests.get(home_logo_url)
    home_image_bytes = io.BytesIO(home_logo_response.content)
    home_img = Image.open(home_image_bytes)

    home_prob_payout_var = round(game_df_pred['home_payout_prob_adjusted'][ind],2)
    away_prob_payout_var = round(game_df_pred['away_payout_prob_adjusted'][ind],2)

    with col1:
        st.image(home_img)
        st.write("Home Team:", game_df_pred['home'][ind])
        st.write("Home Team Spread: ", game_df_pred['odds.details'][ind])
        st.write("Home Team Money Line:", game_df_pred['odds.home.moneyLine'][ind])
        st.write("Home Team Money Line Payout:", round(game_df_pred['home_payout_moneyline'][ind],2))

        if game_df_pred['home_payout_prob_adjusted'][ind] > game_df_pred['away_payout_prob_adjusted'][ind]:
            st.markdown(":white_check_mark: Best Probability Adjusted Payout: $" + str(home_prob_payout_var),  unsafe_allow_html=True)
            # st.write("Rank of Profitability: " + str(game_df_pred['best_payout_rank'][ind]))
        else:
            st.markdown(" Best Probability Adjusted Payout: $" + str(home_prob_payout_var),  unsafe_allow_html=True)
        st.write("Model Predicts...", round(game_df_pred['home_prob'][ind]*100, 2), "% chance of winning")

        # create Shap waterfall plot for home team using shap_values[0][ind] for shap values
        st.pyplot(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][ind], stats_all.iloc[0, :]), )
        st.write("                         ")
        st.write("                         ")

        st.write("-------------------------")

    with col2:
        st.image(away_img)
        st.write("Away Team:", game_df_pred['away'][ind])
        st.write("Away Team Spread: ", game_df_pred['odds.details'][ind])
        st.write("Away Team Money Line:", game_df_pred['odds.away.moneyLine'][ind])
        st.write("Away Team Money Line Payout", round(game_df_pred['away_payout_moneyline'][ind], 2))

        if game_df_pred['home_payout_prob_adjusted'][ind] < game_df_pred['away_payout_prob_adjusted'][ind]:
            st.markdown(":white_check_mark: Best Probability Adjusted Payout: $" + str(away_prob_payout_var),  unsafe_allow_html=True)
            # st.write("Rank of Profitability: " + str(game_df_pred['best_payout_rank'][ind]))
        else:
            st.markdown(" Best Probability Adjusted Payout: $" + str(away_prob_payout_var),  unsafe_allow_html=True)
        st.write("Model Predicts...", round(game_df_pred['away_prob'][ind]*100, 2), "% chance of winning")

        # create Shap waterfall plot for home team using shap_values[0][ind] for shap values
        st.pyplot(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][ind], stats_all.iloc[0, :]), )
        st.write("")
        st.write("-------------------------")

    # # create Shap waterfall plot for home team using shap_values[0][ind] for shap values
    # st.write("SHAP Values for Home Team Probability...")
    # # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][ind], stats_all.iloc[ind, :])
    # st.pyplot(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][ind], stats_all.iloc[0, :]))

    # with col3:
    #     st.write("Money Line:", game_df_pred['odds.details'][ind])
    #     st.write("Over/Under:", game_df_pred['odds.overUnder'][ind])
    #     st.write("Money Line:", game_df_pred['odds.spread'][ind])

    # with col3:
    #     st.write("")
    #     # create Shap waterfall plot for home team using shap_values[0][ind] for shap values
    #     # st.write("SHAP Values for Home Team Probability...")
    #     # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][ind], stats_all.iloc[ind, :])
    #     st.pyplot(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][ind], stats_all.iloc[0, :]), )
    #     st.write("-------------------------")

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

# st.write(game_df_pred['best_payout'].rank(ascending=False))
st.write(game_df_pred)