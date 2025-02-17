import os
import kaggle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import ast

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

def feature_preprocessing(battles_df, winning_card_list_df):
    ############################
    # Notice that some features shouldn't be normalized, such as kingTowerHitPoints,
    # as they are calculated differently according to the trophy level
    ############################
    
    # Normalize features
    scaler = MinMaxScaler()
    features_to_normalize = ['average.startingTrophies', 'loser.startingTrophies', 'winner.startingTrophies',
                             'loser.trophyChange', 'winner.trophyChange']
    battles_df[features_to_normalize] = scaler.fit_transform(battles_df[features_to_normalize])
    features_to_normalize.remove('average.startingTrophies')
    battles_df[features_to_normalize] = battles_df[features_to_normalize] + 1

    # One-hot encode categorical variables
    features_to_onehot = ['arena.id', 'gameMode.id']
    for feature in features_to_onehot:
        battles_df[feature] = pd.get_dummies(battles_df[feature]).idxmax(axis=1).astype('category').cat.codes

    #Imputation
    for feature in battles_df.columns:
        if feature in ['winner.clan.badgeId','loser.clan.badgeId', 'winner.clan.tag', 'loser.clan.tag', 'loser.kingTowerHitPoints']:
           _handle_missing_values(battles_df, feature, strategy='fill')
        elif 'princessTowersHitPoints' in feature:
            _handle_missing_values(battles_df, feature, strategy='fill', value="[0]")

    # Feature engineering
    battles_df = _feature_engineering(battles_df, winning_card_list_df)

    # features to remove
    levels_and_ids = [f'loser.card{i}.id' for i in range(1, 9)] + [f'loser.card{i}.level' for i in range(1, 9)] + \
                        [f'winner.card{i}.id' for i in range(1, 9)] + [f'winner.card{i}.level' for i in range(1, 9)]
    features_to_remove = ['tournamentTag'] + levels_and_ids
    battles_df.drop(columns=features_to_remove, inplace=True)


    return battles_df

def compute_deck_strength(battles_df, card_win_rates):
    deck_strength = np.zeros(len(battles_df))
    for i in range(1, 9):
        card_ids = battles_df[f'winner.card{i}.id']
        card_levels = battles_df[f'winner.card{i}.level']
        win_rates = card_ids.map(card_win_rates).fillna(0.5)  # Default win rate 50% if not seen
        deck_strength += win_rates * card_levels
    return deck_strength

def _feature_engineering(battles_df, winning_card_list_df):
    battles_df['battleTime'] = pd.to_datetime(battles_df['battleTime'])
    numeric_cols = [
        'winner.startingTrophies',
        'loser.startingTrophies',
        'winner.trophyChange',
        'loser.trophyChange',
        'winner.elixir.average',
        'loser.elixir.average'
    ]
    
    battles_df[numeric_cols] = battles_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    battles_df['deck_elixir_variability'] = battles_df[['winner.elixir.average', 'loser.elixir.average']].std(axis=1)
    battles_df['winner_trophy_eff'] = battles_df['winner.trophyChange'] / battles_df['winner.startingTrophies']
    battles_df['loser_trophy_eff'] = battles_df['loser.trophyChange'].abs() / battles_df['loser.startingTrophies']
    winner_card_levels = [f'winner.card{i}.level' for i in range(1,9)]
    loser_card_levels = [f'loser.card{i}.level' for i in range(1,9)]
    battles_df['winner_card_level_std'] = battles_df[winner_card_levels].std(axis=1)
    battles_df['loser_card_level_std'] = battles_df[loser_card_levels].std(axis=1)
    battles_df['winner_spell_troop_ratio'] = (battles_df['winner.spell.count'] + 1) / (battles_df['winner.troop.count'] + 1)
    battles_df['loser_spell_troop_ratio'] = (battles_df['loser.spell.count'] + 1) / (battles_df['loser.troop.count'] + 1)
    battles_df['elixir_gap'] = battles_df['winner.elixir.average'] - battles_df['loser.elixir.average']
    rarities = ['common', 'rare', 'epic', 'legendary']
    battles_df['winner_rarity_diversity'] = battles_df[[f'winner.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    battles_df['loser_rarity_diversity'] = battles_df[[f'loser.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    battles_df['winner.princessTowersHitPoints'] = battles_df['winner.princessTowersHitPoints'].apply(lambda x: sum(ast.literal_eval(x)) if pd.notna(x) and x != '' and x !='[]' else 0)
    battles_df['loser.princessTowersHitPoints'] = battles_df['loser.princessTowersHitPoints'].apply(lambda x: sum(ast.literal_eval(x)) if pd.notna(x) and x != '' and x !='[]' else 0)
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
    # factorized_mappings will save mappings of string columns into indexes in case we need to reverse the factorization
    factorized_mappings = {}
    for col in battles_df.columns:
        if battles_df[col].dtype == 'object':  
            battles_df[col], unique_values = pd.factorize(battles_df[col])  
            factorized_mappings[col] = unique_values 
    winner_counts = battles_df["winner.tag"].value_counts()
    loser_counts = battles_df["loser.tag"].value_counts()
    battles_df["winner_count"] = battles_df["winner.tag"].map(winner_counts)
    battles_df["winner_losing_count"] = battles_df["winner.tag"].map(loser_counts).fillna(0)
    battles_df["total_games_for_winner"] = battles_df["winner_count"] + battles_df["winner_losing_count"]
    battles_df["winner.win_lose_ratio"] = battles_df.apply(lambda row: 1.0 if row["winner_losing_count"] == 0 else row["winner_count"] / row["total_games_for_winner"], axis=1)
    winning_card_set = set(winning_card_list_df["card_id"])
    battles_df["winner_winning_card_count"] = battles_df.apply(lambda row: _count_winning_cards(row, "winner", winning_card_set), axis=1)
    battles_df["loser_winning_card_count"] = battles_df.apply(lambda row: _count_winning_cards(row, "loser", winning_card_set), axis=1)
    # Create ordered list features for winner and loser cards
    battles_df["winner_card_set"] = battles_df.apply(lambda row: tuple(sorted([row[f"winner.card{i}.id"] for i in range(1, 9)])), axis=1)
    battles_df["loser_card_set"] = battles_df.apply(lambda row: tuple(sorted([row[f"loser.card{i}.id"] for i in range(1, 9)])), axis=1)
    # integrate the winner deck card levels into a few informative score features
    battles_df['avg_card_level'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].mean(axis=1)
    battles_df['max_card_level'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].max(axis=1)
    battles_df['min_card_level'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].min(axis=1)
    battles_df['level_variance'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].var(axis=1)
    winner_card_id_cols = [f'winner.card{i}.id' for i in range(1, 9)]
    loser_card_id_cols = [f'loser.card{i}.id' for i in range(1, 9)]
    winner_card_level_cols = [f'winner.card{i}.level' for i in range(1, 9)]
    loser_card_level_cols = [f'loser.card{i}.level' for i in range(1, 9)]
    card_stats = {}
    for i in range(1, 9):
        winner_pairs = list(zip(battles_df[winner_card_id_cols[i-1]], battles_df[winner_card_level_cols[i-1]]))
        loser_pairs = list(zip(battles_df[loser_card_id_cols[i-1]], battles_df[loser_card_level_cols[i-1]]))
        for pair in winner_pairs:
            if pair not in card_stats:
                card_stats[pair] = {'wins': 0, 'appearances': 0}
            card_stats[pair]['wins'] += 1  
            card_stats[pair]['appearances'] += 1  
        for pair in loser_pairs:
            if pair not in card_stats:
                card_stats[pair] = {'wins': 0, 'appearances': 0}
            card_stats[pair]['appearances'] += 1  
    card_win_rates = {pair: stats['wins'] / stats['appearances'] for pair, stats in card_stats.items()}
    battles_df = battles_df.round(6)
    battles_df['deck_weighted_strength'] = compute_deck_strength(battles_df, card_win_rates)
    features_to_normalize = [
    "deck_weighted_strength",
    "avg_card_level",
    "max_card_level",
    "min_card_level",
    "level_variance",
    "winner.elixir.average",
    "synergy_score"
    ]
    scaler = MinMaxScaler()
    battles_df["synergy_score"] = battles_df.apply(_compute_synergy_score, axis=1)
    battles_df[features_to_normalize] = scaler.fit_transform(battles_df[features_to_normalize])
    battles_df["winner_deck_final_score"] = (
        0.35 * battles_df["deck_weighted_strength"] +
        0.20 * battles_df["avg_card_level"] +
        0.15 * (battles_df["max_card_level"] + battles_df["min_card_level"]) / 2 +
        0.10 * (1 - battles_df["level_variance"]) +
        0.10 * np.log1p(battles_df["winner.elixir.average"]) +  # log1p(x) = log(1 + x) for stability
        0.10 * battles_df["synergy_score"]
    )
    battles_df = battles_df.round(6)
    #features_to_normalize.remove("deck_weighted_strength")
    #battles_df.drop(columns=features_to_normalize, inplace=True)
    return battles_df

def _count_winning_cards(row, prefix, winning_card_set):
    return sum(row[f"{prefix}.card{i}.id"] in winning_card_set for i in range(1, 9))

def _compute_synergy_score(row):
    """Calculates synergy based on rarity diversity and spell/troop balance."""
    buildings = row["winner.structure.count"]
    rarity_diversity = row["winner_rarity_diversity"]
    if buildings < 3:
        balance_anchor = 0
        if buildings == 0:
            balance_anchor = 0.355
        elif buildings == 1:
            balance_anchor = 0.44
        else:
            balance_anchor = 0.375
        penalty = abs(row["winner_spell_troop_ratio"] - balance_anchor)
        return rarity_diversity * (1 - penalty)
    else:
        return rarity_diversity * row["winner_spell_troop_ratio"]

def _handle_missing_values(df, column, strategy='auto', n_neighbors=5, value=-1):
    """
    Handle missing values based on each column's characteristics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if df[column].isnull().sum() == 0:
        return df  
    if strategy == 'auto':
        if pd.api.types.is_numeric_dtype(df[column]):
            # Use median if skewed, otherwise use mean
            if abs(df[column].skew()) > 1:
                df[column].fillna(df[column].median(), inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)
        else:
            # Use mode for categorical columns
            df[column].fillna(df[column].mode()[0], inplace=True)
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[column] = imputer.fit_transform(df[[column]]).flatten()
    elif strategy == 'fill':
        df[column] = df[column].fillna(value)
    elif strategy == 'drop':
        df = df.dropna(subset=[column])
    elif strategy == 'fill_zero':
        df[column] = df[column].fillna(0)
    elif strategy == 'fill_mean':
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = df[column].fillna(df[column].mean())
    elif strategy == 'fill_median':
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = df[column].fillna(df[column].median())
    elif strategy == 'fill_mode':
        df[column] = df[column].fillna(df[column].mode()[0])
    elif strategy == 'interpolate':
        df[column] = df[column].interpolate()
    elif strategy == None:
        pass
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df

def get_numerical_dataset(battles_df):
    numerical_columns = battles_df.select_dtypes(include=[np.number]).columns
    id_features_to_remove = ['winner.tag', 'loser.tag', 'gameMode.id', 'winner.clan.tag', 'winner.clan.badgeId', \
                            'loser.clan.badgeId', 'loser.clan.tag']
    df_numeric = battles_df[numerical_columns].copy()
    df_numeric = df_numeric.drop(columns=id_features_to_remove)
    return df_numeric

def get_pca_optimal_components(battles_df):
    df_numeric = get_numerical_dataset(battles_df)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    pca = PCA().fit(df_scaled)  # Compute PCA
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n = np.argmax(cumulative_variance >= 0.95) + 1
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), cumulative_variance, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=0.95, color='r', linestyle='--')  # 95% threshold heuristic
    plt.show()
    print(f"Best component: {n} with a cumulative_variance value of: {cumulative_variance[n-1]:.4f}")
    return df_scaled, n