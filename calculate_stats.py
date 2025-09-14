import json
import requests
import time
import os                 # 1. Import the 'os' module to access environment variables
from dotenv import load_dotenv # 2. Import the function to load the .env file

load_dotenv()             # 3. Load variables from the .env file

# --- CONFIGURATION ---
# IMPORTANT: The API key is now loaded from your .env file
API_KEY = os.getenv("API_KEY") 
# The two API regions we need (e.g., 'euw1' for player info, 'europe' for matches)
REGION_API = "euw1" 
REGION_MATCHES = "europe"
# Path to your DDragon file
DDRAGON_TACTICIANS_FILE = "tft-tactician.json"

# --- CHANGED ---
# Our starting point is now a dictionary with the name as the key and PUUID as the value.
# You would populate this from your database.
PLAYERS_DATA_FROM_DB = {
    "Numb2r3": "w_U8sPX4KleHEWTVczpToW38HDpSY8w4fIL9oGM3qXW0_MJf0nZHIFfkAKpfe9SzIoqfGyRPhWntpQ",
    "Schmiery": "LjJPc8w_khaC0YEEpZocKpnQUmpI-me6MxGKA9oGepmgxNSJxohmQet1jmX2hJS5fXgcr6EyDh0BVg"
}

# --- FUNCTION DEFINITIONS ---

def get_last_match_id(puuid):
    """Fetches the ID of the most recent TFT match for a given PUUID."""
    if not puuid: return None
    # 'count=1' gets only the most recent match
    api_url = f"https://{REGION_MATCHES}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids?count=1&api_key={API_KEY}"
    response = requests.get(api_url)
    if response.status_code == 200 and response.json():
        return response.json()[0] # The response is a list, we want the first item
    else:
        print(f"Error fetching match history for PUUID {puuid}: {response.status_code}")
        return None

def get_companion_item_id_from_match(match_id, puuid):
    """Gets the companion's item_ID from a specific match for a specific player."""
    if not match_id or not puuid: return None
    api_url = f"https://{REGION_MATCHES}.api.riotgames.com/tft/match/v1/matches/{match_id}?api_key={API_KEY}"
    response = requests.get(api_url)
    if response.status_code == 200:
        match_data = response.json()
        # We need to find our player in the list of participants
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                # We now return the item_ID, which corresponds to the ID in the JSON file
                return participant['companion']['item_ID']
    else:
        print(f"Error fetching details for match {match_id}: {response.status_code}")
        return None

def find_legend_image_name(item_id_from_api, ddragon_data):
    """Finds the image filename from the DDragon data using the companion's item ID."""
    if not item_id_from_api: return "default.png"
    
    # The item ID from the API is an integer, but the keys in the JSON are strings.
    item_id_str = str(item_id_from_api)

    # We can now do a direct lookup instead of looping, which is much faster.
    if item_id_str in ddragon_data['data']:
        return ddragon_data['data'][item_id_str]['image']['full']
            
    print(f"Warning: Could not find image for item ID: {item_id_str}")
    return "default.png" # Return a default if not found

# --- MAIN SCRIPT LOGIC ---

# 1. Load the DDragon data into memory once
try:
    with open(DDRAGON_TACTICIANS_FILE, encoding='utf-8') as f:
        ddragon_tacticians_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find the DDragon file at: {DDRAGON_TACTICIANS_FILE}")
    exit()

all_players_card_data = []

# 2. Loop through each player in our new dictionary
for player_name, player_puuid in PLAYERS_DATA_FROM_DB.items():
    print(f"Processing data for: {player_name}...")
    
    last_match = get_last_match_id(player_puuid)
    time.sleep(0.5) # Small delay

    # This variable now holds the item ID (e.g., 27034)
    companion_item_id = get_companion_item_id_from_match(last_match, player_puuid)
    
    # Get the image filename from our local DDragon file using the item ID
    image_filename = find_legend_image_name(companion_item_id, ddragon_tacticians_data)

    # 3. Construct the final data for this player's card
    player_card_data = {
        "name": player_name,
        "imageUrl": f"ddragon/15.18.1/img/tft-tactician/{image_filename}",
        "stats": {
            "Win Rate": "0%", "Avg. Placement": "0", "Top 4 Rate": "0%",
            "Games Played": "0", "Current LP": "0", "Peak Rank": "Unranked"
        },
        "performance": []
    }
    
    all_players_card_data.append(player_card_data)
    print(f" -> Found Little Legend: {image_filename}")
    
    time.sleep(1) 

# 4. Write the final, complete data to players.json
with open('players.json', 'w') as f:
    json.dump(all_players_card_data, f, indent=4)

print("\nSuccessfully created players.json with dynamic Little Legend images!")