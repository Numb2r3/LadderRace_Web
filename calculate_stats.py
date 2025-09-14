import json

# In the future, you can do all your complex calculations here.
# For now, we will just define the final data structure.

players_data = [
    {
        "name": "ProGamer123",
        "imageUrl": "https://placehold.co/400x400/1F2937/FFFFFF?text=Ao+Shin",
        "stats": {
            "Win Rate": "62%",
            "Avg. Placement": "3.2",
            "Top 4 Rate": "78%",
            "Games Played": "112",
            "Current LP": "850",
            "Peak Rank": "Master"
        },
        "performance": [4, 2, 5, 1, 3, 2, 1, 4, 6, 2, 3, 1]
    },
    {
        "name": "TFTMastermind",
        "imageUrl": "https://placehold.co/400x400/1F2937/FFFFFF?text=Choncc",
        "stats": {
            "Win Rate": "55%",
            "Avg. Placement": "3.8",
            "Top 4 Rate": "71%",
            "Games Played": "98",
            "Current LP": "620",
            "Peak Rank": "Diamond I"
        },
        "performance": [8, 5, 3, 2, 4, 1, 3, 5, 2, 4, 6, 3]
    },
    {
        "name": "PixelPoro",
        "imageUrl": "https://placehold.co/400x400/1F2937/FFFFFF?text=Poro",
        "stats": {
            "Win Rate": "58%",
            "Avg. Placement": "3.5",
            "Top 4 Rate": "75%",
            "Games Played": "150",
            "Current LP": "780",
            "Peak Rank": "Master"
        },
        "performance": [3, 1, 4, 1, 5, 2, 6, 2, 3, 1, 4, 2]
    }
]

# This is the key part: writing the data to a JSON file.
# The 'indent=4' makes the file readable for humans.
with open('players.json', 'w') as f:
    json.dump(players_data, f, indent=4)

print("Successfully created players.json!")