# nfl_statistics

I want to do analysis on NFL game statistics to understand if game stats have any predicting power towards which team wins the game.

## Table of Contents

- [Environment](#environment)
- [Project Structure](#project-structure)

## Environment

- [Python Environment](https://www.python.org/downloads/release/python-3111/)
- [Requirements File](requirements.txt)

  - in your python environment simply run `pip install -r requirements.txt` after cloning the repo in your command line

- [Streamlit App = Hosted on public streamlit community](https://lawson-roberts-nfl-statistics-app-20jqx8.streamlit.app/)
- if you would like to run locally to develop and make changes of your own just run the following after cloning the repo in your command line `streamlit run app.py`

## Project Structure

- Folders
  - [.gitub](.github/) - coming soon. This is where all the Github CI yaml files / config lives.
  - [CI Jobs](ci_jobs/) - coming soon. This is where all the Github CI jobs that will be getting executed.
  - [data](data/) - includes raw data as well as data that is ready for the model to train on and predict.
  - [models](models/) - There were a variety of models that I have tried, all with similar performance. All the pickled files live here.
    - Currently running this random forest model [models](models/random_forest_model_1.sav) using paramters shown here
    - `python rf_clf =RandomForestClassifier(max_depth = 10,n_estimators=40, max_features = 'log2', random_state=0)`
  - [predictions](predictions/) - Holding model prediction files. These are used to score how the model is performing in production.
- Files
  - [Cleaning](cleaning_nfl_dataset.ipynb) - Exploring how I could clean this data and make transformation to be machine learning ready.
  - [Classification Explore](nfl_dataset_exploration_classification.ipynb) - this is where I started. Later I realized I wouldn't have all the same features so pivoted to the notebook below `prediction_pipeline_exploration.ipynb`.
  - [Prediction Pipeline Testing](prediction_pipeline_exploration.ipynb) - This was exploration on the data I would have avaialble at the time of prediction.
  - [Pipeline](pipeline_in_order.ipynb) - This was exploration on what steps I needed for the CI jobs I will be working on next to automate training and collecting new data.
  - [Data Collection - Web Scraping](scrap_nfl_game_stats_explore.ipynb) - This was exploration of how I could collect this data to train my model.
