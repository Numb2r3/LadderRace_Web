# =============================================================================
# JSON CREATOR SCRIPT
# =============================================================================
# This script reads the intermediate `processed_data.json` file, formats the
# data into a user-friendly structure for the webpage, and saves the final
# `players.json` that will be used by the playercards.html file.
# =============================================================================

import json
import pandas as pd

# --- Configuration ---
INPUT_FILE = "processed_data.json"
OUTPUT_FILE = "players.json"
DDRAGON_IMG_PATH = "ddragon/15.18.1/img/tft-tactician/"
DDRAGON_REGALIA_PATH = "ddragon/15.18.1/img/tft-regalia/" # Path to rank emblems

def format_rank(tier, division, lp):
    """Formats rank components into a readable string."""
    if not tier or pd.isna(tier) or tier == 'UNRANKED':
        return "Unranked"
    if tier in ['MASTER', 'GRANDMASTER', 'CHALLENGER']:
        return f"{tier.title()} {int(lp)} LP"
    return f"{tier.title()} {division} {int(lp)} LP"

def main():
    """
    Loads processed data, formats it for the playercards,
    and saves the final players.json.
    """
    # Step 1: Load the processed data from the first script
    print(f"--- Loading data from {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE}. Please run data_processor.py first.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not read {INPUT_FILE}. The file might be empty or corrupted.")
        return

    players_df = pd.DataFrame(data)
    print(f"Loaded data for {len(players_df)} players.")

    # Step 2: Prepare the final list for the playercards
    final_player_data = []
    
    print("\n--- Formatting data for playercards ---")
    for index, player in players_df.iterrows():
        # Calculate Top 4 Rate
        total_games = player.get('total_games', 0)
        wins = player.get('wins', 0)
        top_4_rate = f"{round((wins / total_games) * 100)}%" if total_games > 0 else "0%"

        # Format rank strings
        current_rank_str = format_rank(player.get('tier'), player.get('division'), player.get('league_points'))
        peak_rank_str = format_rank(player.get('peak_tier'), player.get('peak_division'), player.get('peak_lp'))
        
        # Get Rank Emblem URLs
        final_tier = player.get('tier', 'UNRANKED').title()
        peak_tier = player.get('peak_tier', 'UNRANKED').title()
        
        final_rank_icon_url = f"{DDRAGON_REGALIA_PATH}TFT_Regalia_{final_tier}.png" if final_tier != 'Unranked' else ""
        peak_rank_icon_url = f"{DDRAGON_REGALIA_PATH}TFT_Regalia_{peak_tier}.png" if peak_tier != 'Unranked' else ""

        # Build the dictionary of stats for the card
        stats_dict = {
            "Position": str(player.get('position', 'N/A')),
            "Final Rank": current_rank_str,
            "Peak Rank": peak_rank_str,
            "Top 4 Rate": top_4_rate,
            "Games Played": str(player.get('total_games', 0)),
            "LP Gained": str(player.get('lp_gained', 0)),
            "LP / Game": str(player.get('lp_gain_per_game', '0.0')),
            "LP / Day": str(player.get('lp_gain_per_day', '0.0'))
        }

        # Build the final dictionary for this player
        player_card = {
            "name": player.get('display_name'),
            "imageUrl": f"{DDRAGON_IMG_PATH}{player.get('image_filename')}",
            "graphUrl": player.get('graph_image_path'),
            "finalRankIconUrl": final_rank_icon_url,
            "peakRankIconUrl": peak_rank_icon_url,
            "stats": stats_dict
        }
        final_player_data.append(player_card)
    
    # Step 3: Save the final players.json file
    print(f"\n--- Saving final data to {OUTPUT_FILE} ---")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_player_data, f, indent=4)
        
    print(f"Successfully created {OUTPUT_FILE} for the playercards webpage.")

if __name__ == "__main__":
    main()