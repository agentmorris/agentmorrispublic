#%% Imports 

import os
import time

from tqdm import tqdm
from sportsipy.nfl.teams import Teams


#%% Constants

year = 2009
sleep_time_per_request = 15
output_folder = r'g:\temp\nfl-game-ranks'

    
#%% Retrieve teams

teams = Teams(2009)
assert len(teams) == 32

print('Retrieved {} teams:\n'.format(len(teams)))
for t in teams:
    print(t)

    
#%% Retrieve schedules    

team_abbreviation_to_schedule = {}
team_abbreviation_to_name = {}

for i_team,t in tqdm(enumerate(teams),total=len(teams)):
    abbr = t.abbreviation
    schedule = t.schedule
    team_abbreviation_to_schedule[abbr] = schedule
    team_abbreviation_to_name[abbr] = t.name
    if i_team != (len(teams) - 1):
        time.sleep(sleep_time_per_request)


#%% Retrieve box scores and schedule details

from collections import defaultdict

team_abbreviation_to_game_details = defaultdict(list)

# abbr = next(iter(team_abbreviation_to_schedule.keys()))
for abbr in team_abbreviation_to_schedule.keys():
    
    print('Retrieving box scores for {}'.format(abbr))
    schedule = team_abbreviation_to_schedule[abbr]
    
    # game = schedule[0]
    for game in tqdm(schedule):
        
        game_details = {}
        game_details['datetime'] = game.datetime
        game_details['result'] = game.result
        game_details['overtime'] = game.overtime
        game_details['opponent_abbr'] = game.opponent_abbr
        game_details['points_scored'] = game.points_scored
        game_details['points_allowed'] = game.points_allowed

        game_details['dataframe'] = game.dataframe
        game_details['dataframe_extended'] = game.dataframe_extended
        
        team_abbreviation_to_game_details[abbr].append(game_details)
        time.sleep(sleep_time_per_request)
        
    # ...for each game
    
# ...for each team


#%% Serialize results

os.makedirs(output_folder,exist_ok=True)
output_file = os.path.join(output_folder,'sportsipy_results.pickle')

sportsipy_results = {}

sportsipy_results['team_abbreviation_to_schedule'] = team_abbreviation_to_schedule
sportsipy_results['team_abbreviation_to_name'] = team_abbreviation_to_name
sportsipy_results['team_abbreviation_to_game_details'] = team_abbreviation_to_game_details

import pickle
with open(output_file,'wb') as f:
    pickle.dump(sportsipy_results,f)


#%% Deserialize results

with open(output_file,'rb') as f:
    sportsipy_results = pickle.load(f)

