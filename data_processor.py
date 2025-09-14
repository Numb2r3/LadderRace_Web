# =============================================================================
# DATA PROCESSING SCRIPT
# =============================================================================
# This script fetches player data from a database, enriches it with data
# from the Riot API, generates performance graphs, and saves the combined
# result to an intermediate JSON file (`processed_data.json`).
# This script is the final Python version of the logic developed in the
# test.ipynb notebook.
# =============================================================================

# Step 1: Setup and Imports
import json
import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
import sqlalchemy
from sqlalchemy import text
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from adjustText import adjust_text

load_dotenv()

# Step 2: Configuration
API_KEY = os.getenv("API_KEY")
REGION_MATCHES = "europe"
DDRAGON_TACTICIANS_FILE = "tft-tactician.json"
OUTPUT_FILE = "processed_data.json"
GRAPH_OUTPUT_DIR = "graphs" # Folder to save graph images

# =============================================================================
# Step 3: Function Definitions
# =============================================================================

# --- DATABASE FUNCTIONS ---
def get_sql_config():
    """Loads database credentials from the .env file."""
    needed_keys = ['host', 'port', 'dbname','user','password']
    dotenv_dict = dotenv_values(".env")
    return {key:dotenv_dict[key] for key in needed_keys if key in dotenv_dict}

def get_engine_alchemy():
    """Creates a SQLAlchemy engine for a PostgreSQL database."""
    config = get_sql_config()
    if not all(key in config for key in ['user', 'password', 'host', 'port', 'dbname']):
        raise KeyError("One or more required database keys are missing in your .env file")
    connection_url = sqlalchemy.URL.create(
        drivername="postgresql+psycopg2",
        username=config['user'],
        password=config['password'],
        host=config['host'],
        port=config['port'],
        database=config['dbname']
    )
    return sqlalchemy.create_engine(connection_url)

def get_dataframe(query, params=None):
    """Executes a SQL query and returns the result as a pandas DataFrame."""
    engine = get_engine_alchemy()
    return pd.read_sql_query(sql=text(query), con=engine, params=params)

# --- RIOT API FUNCTIONS ---
def get_last_match_id(puuid):
    """Fetches the most recent match ID for a given PUUID."""
    if not puuid: return None
    api_url = f"https://{REGION_MATCHES}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids?count=1&api_key={API_KEY}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        if response.json():
            return response.json()[0]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching match history for PUUID {puuid}: {e}")
    return None

def get_companion_item_id_from_match(match_id, puuid):
    """Extracts the companion (Little Legend) item ID from a specific match."""
    if not match_id or not puuid: return None
    api_url = f"https://{REGION_MATCHES}.api.riotgames.com/tft/match/v1/matches/{match_id}?api_key={API_KEY}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        match_data = response.json()
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                return participant.get('companion', {}).get('item_ID')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for match {match_id}: {e}")
    return None

def find_legend_image_name(item_id, ddragon_data):
    """Finds the image filename in the DDragon data using the item_ID."""
    if not item_id: return "default.png"
    item_id_str = str(item_id)
    tactician_info = ddragon_data.get('data', {}).get(item_id_str)
    if tactician_info:
        return tactician_info.get('image', {}).get('full', 'default.png')
    print(f"Warning: Could not find image for item ID: {item_id_str}")
    return "default.png"

# --- DATA ENRICHMENT FUNCTION ---
def enrich_with_legend_data(player_df, ddragon_data):
    """
    Fetches Little Legend data for each player and adds it to the DataFrame.
    This function contains all the Riot API calls.
    """
    legend_images = []
    print("\n--- Enriching with Riot API data ---")
    for index, row in player_df.iterrows():
        puuid = row['puuid']
        print(f"Fetching legend for player: {row['display_name']}...")
        last_match_id = get_last_match_id(puuid)
        time.sleep(1.5)
        item_id = get_companion_item_id_from_match(last_match_id, puuid)
        image_filename = find_legend_image_name(item_id, ddragon_data)
        legend_images.append(image_filename)
        print(f" -> Found: {image_filename}")
        time.sleep(1)
    
    player_df['image_filename'] = legend_images
    return player_df

# --- HELPER FUNCTION FOR RANK SCORE CALCULATION ---
def calculate_rank_scores(df, tier_values, division_values):
    """Calculates a numerical score based on tier, division, and LP."""
    df = df.copy()
    df['tier_score'] = df['tier'].map(tier_values).fillna(-3000)
    df['division_score'] = df['division'].map(division_values).fillna(0)
    is_master_tier = df['tier'].isin(['MASTER', 'GRANDMASTER', 'CHALLENGER'])
    df.loc[is_master_tier, 'division_score'] = 0
    df['score'] = df['tier_score'] + df['division_score'] + df['league_points']
    return df

# --- BENCHMARK CALCULATION FUNCTION ---
def calculate_benchmark_histories(lp_history_df, tier_values, division_values):
    """
    Calculates the score history for the 1st, 10th, and 30th ranked players
    at each unique timestamp using a robust, stateful method.
    """
    if lp_history_df.empty:
        return pd.DataFrame()
    
    history_with_scores = calculate_rank_scores(lp_history_df, tier_values, division_values)
    history_with_scores['retrieved_at'] = pd.to_datetime(history_with_scores['retrieved_at'])

    all_timestamps = sorted(history_with_scores['retrieved_at'].unique())
    all_puuids = history_with_scores['puuid'].unique()
    
    current_scores = pd.Series(index=all_puuids, dtype=float)
    benchmark_data = []

    print("\n--- Calculating leaderboard benchmarks (optimized method) ---")
    for i, ts in enumerate(all_timestamps):
        updates_at_ts = history_with_scores[history_with_scores['retrieved_at'] == ts]
        current_scores.update(pd.Series(updates_at_ts['score'].values, index=updates_at_ts['puuid']))
        
        valid_scores = current_scores.dropna()
        sorted_scores = valid_scores.sort_values(ascending=False).values
        
        score_1 = sorted_scores[0] if len(sorted_scores) >= 1 else None
        score_10 = sorted_scores[9] if len(sorted_scores) >= 10 else None
        score_30 = sorted_scores[29] if len(sorted_scores) >= 30 else None
        
        benchmark_data.append({
            'timestamp': ts, 
            'rank_1_score': score_1, 
            'rank_10_score': score_10, 
            'rank_30_score': score_30
        })
        print(f"Processing benchmarks: {i+1}/{len(all_timestamps)} timestamps complete.", end='\r')
    
    print("\nBenchmark calculation complete.")
    return pd.DataFrame(benchmark_data)


# --- GRAPH PLOTTING FUNCTION ---
def create_player_graph(player_history_df, player_name, benchmark_df, output_path):
    """Generates and saves a performance graph for a single player with benchmarks."""
    if player_history_df.empty:
        print(f"No history for {player_name}, skipping graph.")
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if not benchmark_df.empty:
        sns.lineplot(data=benchmark_df, x='timestamp', y='rank_1_score', ax=ax, color='gold', linewidth=1.5, linestyle='--', label='1st Place', alpha=0.7)
        sns.lineplot(data=benchmark_df, x='timestamp', y='rank_10_score', ax=ax, color='silver', linewidth=1.5, linestyle=':', label='10th Place', alpha=0.7)
        sns.lineplot(data=benchmark_df, x='timestamp', y='rank_30_score', ax=ax, color='#CD7F32', linewidth=1.5, linestyle=':', label='30th Place', alpha=0.7)

    sns.lineplot(data=player_history_df, x='retrieved_at', y='score', ax=ax, color='#a855f7', linewidth=2.5)

    tier_boundaries = [-2800, -2400, -2000, -1600, -1200, -800, -400, 0, 500, 1000, 1500]
    tier_labels = ['Iron', 'Bronze', 'Silver', 'Gold', 'Plat', 'Em', 'Dia', 'Master', '500 LP', '1000 LP', '1500 LP']
    ax.set_yticks(tier_boundaries)
    ax.set_yticklabels(tier_labels)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=15)
    
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.legend(loc='upper left', fontsize='small', frameon=False, labelcolor='white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, transparent=True)
    plt.close()

# =============================================================================
# Step 4: Main Processing Logic
# =============================================================================
def main():
    """Main function to run the data processing workflow."""
    print("Loading DDragon tacticians file...")
    try:
        with open(DDRAGON_TACTICIANS_FILE, encoding='utf-8') as f:
            ddragon_tacticians_data = json.load(f)
        print("DDragon file loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Could not find DDragon file at: {DDRAGON_TACTICIANS_FILE}")
        return

    SERVER_ID_TO_FILTER = "839819217544937513"
    RACE_START_DATE = "2025-07-01 00:00:00"
    CUTOFF_DATE = "2025-09-08 00:05:00"

    latest_stats_query = """
        WITH LatestHistory AS (
            SELECT
                lah.riot_account_id, lah.league_points, lah.tier, lah.division,
                lah.wins, lah.losses,
                ROW_NUMBER() OVER(PARTITION BY lah.riot_account_id ORDER BY lah.retrieved_at DESC) as rn
            FROM public.riot_account_lp_history AS lah
            WHERE lah.queue_type = 'RANKED_TFT' AND lah.retrieved_at <= :cutoff_date_param
        )
        SELECT
            ra.game_name AS display_name, ra.puuid,
            lh.league_points, lh.tier, lh.division, lh.wins, lh.losses
        FROM public.server_players AS sp
        JOIN public.riot_accounts AS ra ON sp.riot_account_id = ra.riot_account_id
        JOIN LatestHistory AS lh ON sp.riot_account_id = lh.riot_account_id
        WHERE sp.server_id = :server_id_param AND lh.rn = 1;
    """
    lp_history_query = """
        SELECT
            ra.puuid,
            lah.league_points,
            lah.tier,
            lah.division,
            lah.wins,
            lah.losses,
            lah.retrieved_at
        FROM public.riot_account_lp_history AS lah
        JOIN public.server_players AS sp ON lah.riot_account_id = sp.riot_account_id
        JOIN public.riot_accounts AS ra ON sp.riot_account_id = ra.riot_account_id
        WHERE sp.server_id = :server_id_param 
          AND lah.queue_type = 'RANKED_TFT' 
          AND lah.retrieved_at BETWEEN :start_date_param AND :cutoff_date_param
        ORDER BY ra.puuid, lah.retrieved_at ASC;
    """
    
    print(f"Fetching player data for server {SERVER_ID_TO_FILTER}...")
    try:
        params = {
            "server_id_param": SERVER_ID_TO_FILTER,
            "start_date_param": RACE_START_DATE,
            "cutoff_date_param": CUTOFF_DATE
        }
        latest_stats_df = get_dataframe(latest_stats_query, params=params)
        lp_history_df = get_dataframe(lp_history_query, params=params)
        print(f"Successfully fetched stats for {len(latest_stats_df)} players and {len(lp_history_df)} total LP entries.")
    except Exception as e:
        print(f"ERROR: Could not fetch from database. Details: {e}")
        return
    
    tier_values = { 'MASTER': 0, 'GRANDMASTER': 0, 'CHALLENGER': 0, 'DIAMOND': -400, 'EMERALD': -800, 'PLATINUM': -1200, 'GOLD': -1600, 'SILVER': -2000, 'BRONZE': -2400, 'IRON': -2800, 'UNRANKED': -3000 }
    division_values = {'IV': 0, 'III': 100, 'II': 200, 'I': 300}

    print("\n--- Calculating player positions and final stats ---")
    if not latest_stats_df.empty:
        latest_stats_df = calculate_rank_scores(latest_stats_df, tier_values, division_values)
        latest_stats_df = latest_stats_df.rename(columns={'score': 'final_score'})
        latest_stats_df = latest_stats_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        latest_stats_df['position'] = latest_stats_df.index + 1
        latest_stats_df['total_games'] = latest_stats_df['wins'] + latest_stats_df['losses']
        print("Positions and final stats calculated.")

    if not lp_history_df.empty:
        history_with_scores = calculate_rank_scores(lp_history_df, tier_values, division_values)
        peak_indices = history_with_scores.groupby('puuid')['score'].idxmax()
        peak_rank_df = history_with_scores.loc[peak_indices][['puuid', 'tier', 'division', 'league_points']]
        peak_rank_df = peak_rank_df.rename(columns={'tier': 'peak_tier', 'division': 'peak_division', 'league_points': 'peak_lp'})
        latest_stats_df = pd.merge(latest_stats_df, peak_rank_df, on='puuid', how='left')
        print("Peak ranks calculated.")
    
    print("\n--- Calculating LP Gained and advanced stats ---")
    if not lp_history_df.empty and 'final_score' in latest_stats_df.columns:
        history_with_scores = calculate_rank_scores(lp_history_df, tier_values, division_values)
        start_indices = history_with_scores.groupby('puuid')['retrieved_at'].idxmin()
        start_rank_df = history_with_scores.loc[start_indices].copy()

        start_rank_df['start_score'] = start_rank_df['score']
        start_rank_df = start_rank_df.rename(columns={'retrieved_at': 'retrieved_at_start'})
        
        latest_stats_df = pd.merge(latest_stats_df, start_rank_df[['puuid', 'start_score', 'wins', 'losses', 'retrieved_at_start']], on='puuid', how='left', suffixes=('', '_start'))
        
        latest_stats_df['lp_gained'] = (latest_stats_df['final_score'] - latest_stats_df['start_score']).fillna(0).astype(int)
        
        games_at_start = (latest_stats_df['wins_start'] + latest_stats_df['losses_start']).fillna(0)
        games_played_in_race = latest_stats_df['total_games'] - games_at_start
        
        cutoff_datetime = datetime.fromisoformat(CUTOFF_DATE)
        latest_stats_df['retrieved_at_start'] = pd.to_datetime(latest_stats_df['retrieved_at_start'])
        days_played = (cutoff_datetime - latest_stats_df['retrieved_at_start']).dt.days + 1
        
        latest_stats_df['lp_gain_per_day'] = (latest_stats_df['lp_gained'] / days_played.where(days_played > 0, 1)).round(2)
        latest_stats_df['lp_gain_per_game'] = (latest_stats_df['lp_gained'] / games_played_in_race.where(games_played_in_race > 0, 1)).round(2)

        print("LP Gained and advanced stats calculated.")
    
    benchmark_df = calculate_benchmark_histories(lp_history_df, tier_values, division_values)

    if not latest_stats_df.empty:
        latest_stats_df = enrich_with_legend_data(latest_stats_df, ddragon_tacticians_data)

    print("\n--- Generating performance graphs ---")
    if not os.path.exists(GRAPH_OUTPUT_DIR):
        os.makedirs(GRAPH_OUTPUT_DIR)
        print(f"Created directory: {GRAPH_OUTPUT_DIR}")

    graph_paths = []
    if not latest_stats_df.empty:
        history_with_scores = calculate_rank_scores(lp_history_df, tier_values, division_values)
        history_with_scores['retrieved_at'] = pd.to_datetime(history_with_scores['retrieved_at'])

        for index, player in latest_stats_df.iterrows():
            player_puuid = player['puuid']
            player_name = player['display_name']
            
            safe_filename = "".join(x for x in player_name if x.isalnum())
            output_path = os.path.join(GRAPH_OUTPUT_DIR, f"{safe_filename}_graph.png")
            
            player_history = history_with_scores[history_with_scores['puuid'] == player_puuid].sort_values(by='retrieved_at')
            
            create_player_graph(player_history, player_name, benchmark_df, output_path)
            graph_paths.append(output_path)
            print(f"Graph for {player_name} saved to {output_path}")

        latest_stats_df['graph_image_path'] = graph_paths
    
    final_df = latest_stats_df
    
    if 'retrieved_at_start' in final_df.columns:
        final_df['retrieved_at_start'] = final_df['retrieved_at_start'].astype(str)

    processed_player_list = final_df.to_dict('records')

    print("\n--- Saving final data ---")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(processed_player_list, f, indent=4)
    
    print(f"Successfully created {OUTPUT_FILE} with combined data and generated graphs.")


if __name__ == "__main__":
    main()