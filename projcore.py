import os
import kaggle
import pandas as pd
import numpy as np

def download_kaggle_main_db(zip = False, tables_amount = 0, force = False):
    tables = [
        "BattlesStaging_01012021_WL_tagged/BattlesStaging_01012021_WL_tagged.csv",
        "CardMasterListSeason18_12082020.csv",
        "Wincons.csv"
    ]
    additional_tables = [
        "BattlesStaging_01022021_WL_tagged/BattlesStaging_01022021_WL_tagged.csv",
        "BattlesStaging_01032021_WL_tagged/BattlesStaging_01032021_WL_tagged.csv",
        "BattlesStaging_01042021_WL_tagged/BattlesStaging_01042021_WL_tagged.csv",
        "battlesStaging_12072020_to_12262020_WL_tagged/battlesStaging_12072020_to_12262020_WL_tagged.csv",
        "battlesStaging_12272020_WL_tagged/battlesStaging_12272020_WL_tagged.csv",
        "BattlesStaging_12292020_WL_tagged/BattlesStaging_12292020_WL_tagged.csv",
        "BattlesStaging_12302020_WL_tagged/BattlesStaging_12302020_WL_tagged.csv",
        "BattlesStaging_12312020_WL_tagged/BattlesStaging_12312020_WL_tagged.csv",
        "battlesStaging_12282020_WL_tagged/battlesStaging_12282020_WL_tagged.csv"
    ]
    script_directory = os.path.dirname(os.path.abspath(__file__))
    for i in range(min(tables_amount, len(additional_tables))):
        tables.append(additional_tables[i])
    for i in range(len(tables)):
        try:
            if not force and os.path.exists(os.path.join(script_directory, tables[i].split('/')[-1])):
                print(f"File {tables[i]} already exists, skipping download")
                continue
            else:
                print("Downloading main dataset")
                print(f"Downloading {tables[i]}")
                kaggle.api.dataset_download_file(
                    "bwandowando/clash-royale-season-18-dec-0320-dataset",
                    path=script_directory,
                    file_name=tables[i]
                )
                print(f"Downloaded and extracted main dataset - table of {tables[i]} to {script_directory}")
        except kaggle.rest.ApiException as e:
            raise ValueError("Kaggle API credentials not found or invalid.") from e
        except Exception as e:
            raise Exception(f"Failed to download main dataset: {e}")
    
def download_kaggle_secondary_db(zip = False, force = False):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        if not force and os.path.exists(os.path.join(script_directory, "clash_royal_data.csv")):
            print(f"clash-royal-data.csv already exists, skipping download")
        else:
            print("Downloading secondary dataset")
            kaggle.api.dataset_download_files(
                "abhinavshaw09/clash-royal-dataset",
                path=script_directory,
                unzip=True
            )
            print(f"Downloaded and extracted secondary dataset to {script_directory}")
    except kaggle.rest.ApiException as e:
        raise ValueError("Kaggle API credentials not found or invalid.") from e
    except Exception as e:
        raise Exception(f"Failed to download secondary dataset: {e}") 
    
def download_kaggle_datasets(zip = False, main_db_tables = 0):
    download_kaggle_main_db(zip, main_db_tables)
    download_kaggle_secondary_db(zip)

def _count_winning_cards(row, prefix, winning_card_set):
    return sum(row[f"{prefix}.card{i}.id"] in winning_card_set for i in range(1, 9))

# Compute rank-based scaling for winner_count, loser_count, and total_count
def _assign_rank(series):
    return pd.qcut(series.rank(method="min"), q=4, labels=[0, 1, 2, 3])

def feature_engineering(battles_df, card_list_df, winning_card_list_df):
    battles_df['battleTime'] = pd.to_datetime(battles_df['battleTime'])
    numeric_cols = [
        'winner.princessTowersHitPoints',
        'loser.princessTowersHitPoints',
        'winner.startingTrophies',
        'loser.startingTrophies',
        'winner.trophyChange',
        'loser.trophyChange',
        'winner.elixir.average',
        'loser.elixir.average'
    ]
    
    for col in numeric_cols:
        battles_df[col] = (
            battles_df[col].astype(str)
            .str.replace(',', '.', regex=False)  # Handle European decimal formats
            .str.replace('[^0-9.]', '', regex=True)  # Remove non-numeric characters
            .replace('', np.nan)  # Convert empty strings to NaN
        )
        battles_df[col] = pd.to_numeric(battles_df[col], errors='coerce')
    
    battles_df['deck_elixir_variability'] = battles_df[['winner.elixir.average', 'loser.elixir.average']].std(axis=1)
    battles_df['winner_trophy_eff'] = battles_df['winner.trophyChange'] / battles_df['winner.startingTrophies']
    battles_df['loser_trophy_eff'] = battles_df['loser.trophyChange'].abs() / battles_df['loser.startingTrophies']
    winner_card_levels = [f'winner.card{i}.level' for i in range(1,9)]
    loser_card_levels = [f'loser.card{i}.level' for i in range(1,9)]
    battles_df['winner_card_level_std'] = battles_df[winner_card_levels].std(axis=1)
    battles_df['loser_card_level_std'] = battles_df[loser_card_levels].std(axis=1)
    battles_df['winner_spell_troop_ratio'] = battles_df['winner.spell.count'] / battles_df['winner.troop.count']
    battles_df['loser_spell_troop_ratio'] = battles_df['loser.spell.count'] / battles_df['loser.troop.count']
    battles_df['elixir_gap'] = battles_df['winner.elixir.average'] - battles_df['loser.elixir.average']
    rarities = ['common', 'rare', 'epic', 'legendary']
    battles_df['winner_rarity_diversity'] = battles_df[[f'winner.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    battles_df['loser_rarity_diversity'] = battles_df[[f'loser.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    battles_df['princess_tower_gap'] = battles_df['winner.princessTowersHitPoints'] - battles_df['loser.princessTowersHitPoints']
    battles_df['win_streak_proxy'] = battles_df['winner.trophyChange'] / 50
    battles_df['winner_has_legendary'] = battles_df['winner.legendary.count'].gt(0).astype(int)
    battles_df['loser_has_legendary'] = battles_df['loser.legendary.count'].gt(0).astype(int)
    battles_df['clan_advantage'] = ((battles_df['winner.clan.tag'].notna()) & (battles_df['loser.clan.tag'].isna())).astype(int)
    battles_df['elixir_advantage'] = battles_df['winner.elixir.average'].gt(battles_df['loser.elixir.average']).astype(int)
    battles_df['balanced_deck_winner'] = ((battles_df['winner.troop.count'] > 2) & (battles_df['winner.spell.count'] > 1) & (battles_df['winner.structure.count'] > 0)).astype(int)
    battles_df['balanced_deck_loser'] = ((battles_df['loser.troop.count'] > 2) & (battles_df['loser.spell.count'] > 1) & (battles_df['loser.structure.count'] > 0)).astype(int)
    arena_mean = battles_df.groupby('arena.id')['winner.totalcard.level'].transform('mean')
    battles_df['underleveled_winner'] = (battles_df['winner.totalcard.level'] < arena_mean).astype(int)
    arena_mean_loser = battles_df.groupby('arena.id')['loser.totalcard.level'].transform('mean')
    battles_df['underleveled_loser'] = (battles_df['loser.totalcard.level'] < arena_mean_loser).astype(int)
    battles_df['crown_dominance'] = battles_df['winner.crowns'].ge(2).astype(int)
    battles_df['tournament_participant'] = battles_df['tournamentTag'].notna().astype(int)
    battles_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in battles_df.columns:
        if battles_df[col].dtype == 'object':
            column_replaced = []
            for s in battles_df[col]:
                if pd.isna(s) or s == '' or not any(char.isdigit() for char in str(s)):
                    column_replaced.append(np.nan)
                else:
                    numeric_str = ""
                    for char in str(s):
                        if char.isdigit() or char == '.':
                            numeric_str += str(ord(char))
                    column_replaced.append(float(numeric_str))
            battles_df[col] = pd.Series(column_replaced)
    winner_counts = battles_df["winner.tag"].value_counts()
    loser_counts = battles_df["loser.tag"].value_counts()
    battles_df["winner_count"] = battles_df["winner.tag"].map(winner_counts)
    battles_df["loser_count"] = battles_df["loser.tag"].map(loser_counts)
    battles_df["total_games"] = battles_df["winner_count"] + battles_df["loser_count"]
    battles_df["win_lose_ratio"] = battles_df.apply(lambda row: 1.0 if row["loser_count"] == 0 else row["winner_count"] / row["loser_count"], axis=1)
    winning_card_set = set(winning_card_list_df["card_id"])
    battles_df["winner_winning_card_count"] = battles_df.apply(lambda row: _count_winning_cards(row, "winner", winning_card_set), axis=1)
    battles_df["loser_winning_card_count"] = battles_df.apply(lambda row: _count_winning_cards(row, "loser", winning_card_set), axis=1)
    battles_df["winner_rank"] = _assign_rank(battles_df["winner_count"])
    battles_df["loser_rank"] = _assign_rank(battles_df["loser_count"])
    battles_df["total_rank"] = _assign_rank(battles_df["total_games"])
    # Create ordered list features for winner and loser cards
    battles_df["winner_card_set"] = battles_df.apply(lambda row: tuple(sorted([row[f"winner.card{i}.id"] for i in range(1, 9)])), axis=1)
    battles_df["loser_card_set"] = battles_df.apply(lambda row: tuple(sorted([row[f"loser.card{i}.id"] for i in range(1, 9)])), axis=1)
    return battles_df

