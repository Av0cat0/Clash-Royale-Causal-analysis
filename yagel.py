import pandas as pd
import numpy as np
import seaborn as sns
import os

import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join('data', 'BattlesStaging_01012021_WL_tagged.csv'))

data_dir = os.path.join(os.getcwd(), 'data')

print(df.head())

print(df.columns)

def add_features(df):
    df['battleTime'] = pd.to_datetime(df['battleTime'])
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
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '.', regex=False)  # Handle European decimal formats
            .str.replace('[^0-9.]', '', regex=True)  # Remove non-numeric characters
            .replace('', np.nan)  # Convert empty strings to NaN
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # =====================================================================
    df['deck_elixir_variability'] = df[['winner.elixir.average', 'loser.elixir.average']].std(axis=1)
    
    df['winner_trophy_eff'] = df['winner.trophyChange'] / df['winner.startingTrophies']
    df['loser_trophy_eff'] = df['loser.trophyChange'].abs() / df['loser.startingTrophies']

    winner_card_levels = [f'winner.card{i}.level' for i in range(1,9)]
    loser_card_levels = [f'loser.card{i}.level' for i in range(1,9)]
    df['winner_card_level_std'] = df[winner_card_levels].std(axis=1)
    df['loser_card_level_std'] = df[loser_card_levels].std(axis=1)
    
    df['winner_spell_troop_ratio'] = df['winner.spell.count'] / df['winner.troop.count']
    df['loser_spell_troop_ratio'] = df['loser.spell.count'] / df['loser.troop.count']
    
    df['elixir_gap'] = df['winner.elixir.average'] - df['loser.elixir.average']
    
    rarities = ['common', 'rare', 'epic', 'legendary']
    df['winner_rarity_diversity'] = df[[f'winner.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    df['loser_rarity_diversity'] = df[[f'loser.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    
    df['princess_tower_gap'] = df['winner.princessTowersHitPoints'] - df['loser.princessTowersHitPoints']
    
    df['win_streak_proxy'] = df['winner.trophyChange'] / 50
    
    df['winner_has_legendary'] = df['winner.legendary.count'].gt(0).astype(int)
    df['loser_has_legendary'] = df['loser.legendary.count'].gt(0).astype(int)
    
    df['clan_advantage'] = ((df['winner.clan.tag'].notna()) & 
                            (df['loser.clan.tag'].isna())).astype(int)
    
    df['elixir_advantage'] = df['winner.elixir.average'].gt(
        df['loser.elixir.average']).astype(int)
    
    df['balanced_deck_winner'] = ((df['winner.troop.count'] > 2) & 
                                 (df['winner.spell.count'] > 1) & 
                                 (df['winner.structure.count'] > 0)).astype(int)
    df['balanced_deck_loser'] = ((df['loser.troop.count'] > 2) & 
                                (df['loser.spell.count'] > 1) & 
                                (df['loser.structure.count'] > 0)).astype(int)
    
    arena_mean = df.groupby('arena.id')['winner.totalcard.level'].transform('mean')
    df['underleveled_winner'] = (df['winner.totalcard.level'] < arena_mean).astype(int)
    arena_mean_loser = df.groupby('arena.id')['loser.totalcard.level'].transform('mean')
    df['underleveled_loser'] = (df['loser.totalcard.level'] < arena_mean_loser).astype(int)
    
    df['crown_dominance'] = df['winner.crowns'].ge(2).astype(int)
    
    df['tournament_participant'] = df['tournamentTag'].notna().astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

added_df = add_features(df)

df.head()

kaggle.api.dataset_download_file(
                "abhinavshaw09/clash-royal-dataset",
                path=data_dir,
                file_name='clash_royal_data.csv'
            )

