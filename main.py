import pandas as pd
import requests

def call_api():
    url = "https://api.sportradar.com/mlb/trial/v8/en/league/injuries.json"
    API_KEY = '...'
    params = {'api_key':API_KEY}
    response = requests.get(url, params=params)
    response = response.json()
    
    return response
    
def flip_name(name):
    last_name = str()
    first_name =str()
    j = 0
    while name[j] != ',':
        last_name += str(name[j])
        j+=1
    for j in range(j +2, len(name)):
        first_name +=str(name[j])
    full_name = f"{first_name} {last_name}"
    return full_name
    
def flip_names(df):
    df['player_name'] = df['player_name'].apply(flip_name)
    return df

def find_injuries(response):
    teams = response["teams"]
    injuries = []
    
    for team in teams:
        team_players = team["players"]
        
        for player in team_players:
            if player["position"] == "P":
                pitcher_name = player["full_name"]
                
                injury_info = player["injuries"]
                pitcher_injury = injury_info
                injury_type = pitcher_injury[0]
                injury_type = injury_type['desc']
                injuries.append({'pitcher_name' : pitcher_name, 'injury_type' : injury_type})
    
    
    return injuries

def get_name(pitcher):
    return(pitcher['pitcher_name'])

def create_total_injuries(df, injury_list):
    injury_column = []

    injured_players = [inj['pitcher_name'] for inj in injury_list]
    injury_types = [inj['injury_type'] for inj in injury_list]
    players_list = df['player_name'].tolist()
    
    i = 0
    j = 0
    
    while i < len(players_list):
        player = players_list[i]
        
        if j < len(injured_players) and player == injured_players[j]:
            injury_column.append(injury_types[j])
            i += 1
            j += 1
        elif j < len(injured_players) and injured_players[j] not in players_list:
            j += 1
        else:
            injury_column.append('0')
            i += 1
    
    return injury_column

            
         

    
df = pd.read_csv("C:/Users/nstru/Downloads/savant_data (1).csv")
df = flip_names(df)
df_sorted = df.sort_values(by = "player_name")
response = call_api()
injuries = find_injuries(response)
sorted_injuries = sorted(injuries, key =get_name)

injury_column = create_total_injuries(df_sorted, sorted_injuries)
injury_df = pd.DataFrame(injury_column)
#print(injury_column)

df_sorted['injuries'] = injury_column
print(len(injury_column))
print(len(df_sorted))

df_sorted.to_csv('df_sorted.csv')


