import os
import kaggle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import ast
from sklearn.manifold import TSNE

PCA_VARIENCE_THRESHOLD = 0.95
ROUNDING_PRECISION = 7

def download_kaggle_main_db(zip = False, tables_amount = 0, force = False):
    """
    Downloads the primary Clash Royale dataset from Kaggle.

    Parameters:
    - zip (bool): Whether to download as a zip file.
    - tables_amount (int): Number of additional tables to download.
    - force (bool): If True, forces re-download even if files exist.

    Returns:
    - Downloads CSV files into the script directory.
    """
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
    for table in tables:
        try:
            if not force and os.path.exists(os.path.join(script_directory, table.split('/')[-1])):
                print(f"File {table} already exists, skipping download")
                continue
            else:
                print("Downloading main dataset")
                print(f"Downloading {table}")
                kaggle.api.dataset_download_file(
                    "bwandowando/clash-royale-season-18-dec-0320-dataset",
                    path=script_directory,
                    file_name=table
                )
                print(f"Downloaded and extracted main dataset - table of {table} to {script_directory}")
        except kaggle.rest.ApiException as e:
            raise ValueError("Kaggle API credentials not found or invalid.") from e
        except Exception as e:
            raise Exception(f"Failed to download main dataset: {e}")


def download_kaggle_secondary_db(zip = False, force = False):
    """
    Downloads the secondary Clash Royale datasets from Kaggle.

    Parameters:
    - zip (bool): Whether to download as a zip file.
    - force (bool): If True, forces re-download even if files exist.

    Returns:
    - Downloads CSV files into the script directory.
    """
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
    """
    Preprocesses battle dataset by:
    1. Normalizing SOME numeric features (not every feature should be normalized, such as kingTowerHitPoints).
    2. Encoding categorical variables.
    3. Handling missing values.
    4. Feature engineering to create new useful features.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset.
    - winning_card_list_df (pd.DataFrame): Dataset containing winning cards.

    Returns:
    - pd.DataFrame: The preprocessed dataset.
    """

    # Initial Outliers Handling
    battles_df = _pre_engineering_outliers_handling(battles_df)

    # Normalize features
    scaler = MinMaxScaler()
    features_to_normalize = ['average.startingTrophies', 'loser.startingTrophies', 'winner.startingTrophies',
                             'loser.trophyChange', 'winner.trophyChange']
    battles_df[["normalized_" + feature for feature in features_to_normalize]] = scaler.fit_transform(battles_df[features_to_normalize]) + 1

    # One-hot encode categorical variables
    features_to_onehot = ['gameMode.id']
    for feature in features_to_onehot:
        battles_df.loc[:, feature] = pd.get_dummies(battles_df[feature]).idxmax(axis=1).astype('category').cat.codes

    #Imputation
    for feature in battles_df.columns:
        if feature in ['winner.clan.badgeId','loser.clan.badgeId', 'winner.clan.tag', 'loser.clan.tag', 'loser.kingTowerHitPoints']:
           _handle_missing_values(battles_df, feature, strategy='fill')
        elif 'princessTowersHitPoints' in feature:
            _handle_missing_values(battles_df, feature, strategy='fill', value="[0]")

    # Apply Feature Engineering
    battles_df = _feature_engineering(battles_df, winning_card_list_df)

    # Features Removal
    levels_and_ids = [f'loser.card{i}.id' for i in range(1, 9)] + [f'loser.card{i}.level' for i in range(1, 9)] + \
                        [f'winner.card{i}.id' for i in range(1, 9)] + [f'winner.card{i}.level' for i in range(1, 9)]
    features_to_remove = levels_and_ids + ['winner.clan.tag', 'winner.clan.badgeId', \
                            'loser.clan.badgeId', 'loser.clan.tag', 'tournamentTag']
    battles_df.drop(columns=features_to_remove, inplace=True)

    return battles_df



def _compute_deck_strength(battles_df, card_win_rates):
    """
    Computes the overall strength of a deck based on individual card win rates and levels.
    This function evaluates the effectiveness of a deck by summing the weighted win rates 
    of the cards in a player's deck, considering both the card level and historical win rates.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset.
    - card_win_rates (dict): A dictionary where keys are card IDs and values are their respective 
      win rates (e.g., {card_id1: 0.55, card_id2: 0.62}).

    Returns:
    - np.ndarray: A NumPy array containing the computed strength of each deck in battles_df.
    """
    deck_strength = np.zeros(len(battles_df))
    for i in range(1, 9):
        card_ids = battles_df[f'winner.card{i}.id']
        card_levels = battles_df[f'winner.card{i}.level']
        win_rates = card_ids.map(card_win_rates).fillna(0.5)  # Default win rate 50% if not seen
        deck_strength += win_rates * card_levels
    return deck_strength


def _pre_engineering_outliers_handling(battles_df):
    """
    Handles initial outliers and noise in the battle dataset, before feature engineering.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset.

    Returns:
    - pd.DataFrame: The cleaned dataset with initial outliers removed.
    """
    non_existing_trophy_values = battles_df['winner.trophyChange'].value_counts()
    non_existing_trophy_indexs = non_existing_trophy_values[non_existing_trophy_values >= 20].index
    battles_df = battles_df[battles_df['winner.trophyChange'].isin(non_existing_trophy_indexs)]
    battles_df = battles_df[(battles_df['winner.elixir.average'] >= 2) &
                             (battles_df['winner.elixir.average'] <= 5.5) &
                             (battles_df['gameMode.id'] != 72000023) & # removing noise
                             (battles_df['arena.id'] == 54000050)] # last arena, the only one that matters

    return battles_df


def _post_engineering_outliers_handling(battles_df, deck_total_games):
    """
    Handles outliers and noise in the battle dataset after feature engineering.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset.

    Returns:
    - pd.DataFrame: The cleaned dataset with outliers and noise removed.
    """
    battles_df = battles_df[(battles_df["winner_card_set"].map(deck_total_games) >= 5) |
                             (battles_df["loser_card_set"].map(deck_total_games) >= 5)]
    return battles_df.drop('Unnamed: 0', axis=1)


def _count_winning_cards(row, prefix, winning_card_set):
    """
    Counts how many cards in a player's deck belong to the predefined set of 'winning' cards.

    Parameters:
    - row (pd.Series): A single row from the battles dataset, representing one battle.
    - prefix (str): Either "winner" or "loser", indicating which player's deck is being evaluated.
    - winning_card_set (set): A set of card IDs that are considered 'winning' cards.

    Returns:
    - int: The number of cards in the player's deck that are part of `winning_card_set`.
    """
    return sum(row[f"{prefix}.card{i}.id"] in winning_card_set for i in range(1, 9))


def _compute_synergy_score(row):
    """
    Computes the synergy score based on deck structure, rarity diversity, and balance.

    Parameters:
    - row (pd.Series): A single row from battles_df.

    Returns:
    - float: Synergy score (0 to 1).
    """
    buildings = row["winner.structure.count"]
    rarity_diversity = row["winner.rarity_diversity"]
    if buildings < 3:
        balance_anchor = 0.375 if buildings == 2 else (0.44 if buildings == 1 else 0.355)
        penalty = abs(row["winner.spell_troop_ratio"] - balance_anchor)
        return rarity_diversity * (1 - penalty)
    return rarity_diversity * row["winner.spell_troop_ratio"]
    

def _handle_missing_values(df, column, strategy='auto', n_neighbors=5, value=-1):
    """
    Handles missing values in a DataFrame based on different strategies.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Column name to process.
    - strategy (str): How to handle missing values.
    - n_neighbors (int): For KNN imputation.
    - value (int/float/str): Value for "fill" strategy.

    Returns:
    - pd.DataFrame: Updated DataFrame with missing values handled.
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


def _elixir_score(elixir_value, lower_bound, upper_bound):
    """
    Computes an elixir efficiency score based on how close the elixir value is to the optimal range.
    The score follows these rules:
    - If `elixir_value` is within [lower_bound, upper_bound], it gets a perfect score of 1.
    - If `elixir_value` is outside the range, it gets penalized based on its distance from the range.
    - The penalty follows an exponential decay function exp(-distance), meaning:
        - Small deviations receive a mild penalty.
        - Large deviations are penalized heavily.

    Parameters:
    - elixir_value (float): The player's average elixir cost in the battle.
    - lower_bound (float): The lower threshold for an optimal elixir value.
    - upper_bound (float): The upper threshold for an optimal elixir value.

    Returns:
    - float: A score between (0,1], where 1 means the elixir cost is optimal.
    """
    if lower_bound <= elixir_value <= upper_bound:
        return 1 
    distance = min(abs(elixir_value - lower_bound), abs(elixir_value - upper_bound))
    return np.exp(-distance)  # Exponential decay to penalize further distances


def _feature_engineering(battles_df, winning_card_list_df):
    """
    Performs feature engineering on the battle dataset to extract meaningful attributes.

    This function:
    1. Converts timestamps to datetime format.
    2. Computes numerical transformations (e.g., ratios, standard deviations).
    3. Derives new features related to elixir usage, deck composition, and card strength.
    4. Encodes categorical variables and normalizes scores.
    5. Computes win-lose ratios and performance metrics.
    6. Removes unnecessary columns to optimize dataset quality.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset containing match statistics.
    - winning_card_list_df (pd.DataFrame): A dataset containing the IDs of strong (high win-rate) cards.

    Returns:
    - pd.DataFrame: The transformed dataset with additional features and cleaned data.
    """
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
    battles_df['winner.trophy_eff'] = battles_df['winner.trophyChange'] / battles_df['winner.startingTrophies']
    battles_df['loser.trophy_eff'] = battles_df['loser.trophyChange'].abs() / battles_df['loser.startingTrophies']
    winner_card_levels = [f'winner.card{i}.level' for i in range(1,9)]
    loser_card_levels = [f'loser.card{i}.level' for i in range(1,9)]
    battles_df['winner.card_level_std'] = battles_df[winner_card_levels].std(axis=1)
    battles_df['loser.card_level_std'] = battles_df[loser_card_levels].std(axis=1)
    battles_df['winner.spell_troop_ratio'] = (battles_df['winner.spell.count'] + 1) / (battles_df['winner.troop.count'] + 1)
    battles_df['loser.spell_troop_ratio'] = (battles_df['loser.spell.count'] + 1) / (battles_df['loser.troop.count'] + 1)
    battles_df['winner_loser.elixir_gap'] = battles_df['winner.elixir.average'] - battles_df['loser.elixir.average']
    rarities = ['common', 'rare', 'epic', 'legendary']
    battles_df['winner.rarity_diversity'] = battles_df[[f'winner.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    battles_df['loser.rarity_diversity'] = battles_df[[f'loser.{r}.count' for r in rarities]].gt(0).sum(axis=1)
    battles_df['winner.princessTowersHitPoints'] = battles_df['winner.princessTowersHitPoints'].apply(lambda x: sum(ast.literal_eval(x)) if pd.notna(x) and x != '' and x !='[]' else 0)
    battles_df['loser.princessTowersHitPoints'] = battles_df['loser.princessTowersHitPoints'].apply(lambda x: sum(ast.literal_eval(x)) if pd.notna(x) and x != '' and x !='[]' else 0)
    battles_df['winnre_loser.princess_tower_gap'] = battles_df['winner.princessTowersHitPoints'] - battles_df['loser.princessTowersHitPoints']
    battles_df['winner.has_legendary'] = battles_df['winner.legendary.count'].gt(0).astype(int)
    battles_df['loser.has_legendary'] = battles_df['loser.legendary.count'].gt(0).astype(int)
    battles_df['winner_loser.elixir_advantage'] = battles_df['winner.elixir.average'].gt(battles_df['loser.elixir.average']).astype(int)
    battles_df['winner.balanced_deck'] = ((battles_df['winner.troop.count'] > 2) & (battles_df['winner.spell.count'] > 1) & (battles_df['winner.structure.count'] > 0)).astype(int)
    battles_df['loser.balanced_deck'] = ((battles_df['loser.troop.count'] > 2) & (battles_df['loser.spell.count'] > 1) & (battles_df['loser.structure.count'] > 0)).astype(int)
    battles_df['winner.crown_dominance'] = battles_df['winner.crowns'].ge(2).astype(int)
    # factorized_mappings will save mappings of string columns into indexes in case we need to reverse the factorization
    # will use it later
    factorized_mappings = {}
    unique_ids = pd.concat([battles_df["winner.tag"], battles_df["loser.tag"]]).unique()
    id_mapping = {val: idx for idx, val in enumerate(unique_ids)}
    battles_df["winner.tag"] = battles_df["winner.tag"].map(id_mapping)
    battles_df["loser.tag"] = battles_df["loser.tag"].map(id_mapping)
    factorized_mappings["winner.tag"] = id_mapping
    for col in battles_df.columns:
        if battles_df[col].dtype == 'object' and col not in ['winner.tag', 'loser.tag']:  
            battles_df[col], unique_values = pd.factorize(battles_df[col])  
            factorized_mappings[col] = unique_values 
    winner_counts = battles_df["winner.tag"].value_counts()
    loser_counts = battles_df["loser.tag"].value_counts()
    battles_df["winner.winning_count"] = battles_df["winner.tag"].map(winner_counts)
    battles_df["winner.losing_count"] = battles_df["winner.tag"].map(loser_counts).fillna(0)
    battles_df["winner.total_games_for"] = battles_df["winner.winning_count"] + battles_df["winner.losing_count"]
    battles_df["winner.win_lose_ratio"] = np.where(
    battles_df["winner.losing_count"] == 0,
    1.0,
    battles_df["winner.winning_count"] / battles_df["winner.total_games_for"]
    )
    battles_df["winner.win_lose_ratio_Z_score"] = (battles_df["winner.win_lose_ratio"] - battles_df["winner.win_lose_ratio"].mean()) / battles_df["winner.win_lose_ratio"].std()
    winning_card_set = set(winning_card_list_df["card_id"])
    battles_df["winner.winning_card_count"] = battles_df.apply(lambda row: _count_winning_cards(row, "winner", winning_card_set), axis=1)
    battles_df["loser.winning_card_count"] = battles_df.apply(lambda row: _count_winning_cards(row, "loser", winning_card_set), axis=1)
    # Create ordered list features for winner and loser cards
    battles_df["winner.card_set"] = battles_df.apply(lambda row: tuple(sorted([row[f"winner.card{i}.id"] for i in range(1, 9)])), axis=1)
    battles_df["loser.card_set"] = battles_df.apply(lambda row: tuple(sorted([row[f"loser.card{i}.id"] for i in range(1, 9)])), axis=1)
    # integrate the winner deck card levels into a few informative score features
    battles_df['winner.avg_card_level'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].mean(axis=1)
    battles_df['winner.max_card_level'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].max(axis=1)
    battles_df['winner.min_card_level'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].min(axis=1)
    battles_df['winner.level_variance'] = battles_df[[f'winner.card{i}.level' for i in range(1, 9)]].var(axis=1)
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
    deck_wins = battles_df.groupby("winner.card_set").size()
    deck_losses = battles_df.groupby("loser.card_set").size()
    deck_total_games = deck_wins.add(deck_losses, fill_value=0)
    deck_win_rate = (deck_wins / deck_total_games).fillna(0)
    battles_df["winner.win_rate"] = battles_df["winner.card_set"].map(deck_win_rate)
    battles_df['winner.high_win_rate']=battles_df['winner.win_rate']>0.75
    battles_df = battles_df.round(ROUNDING_PRECISION)
    battles_df['winner.deck_weighted_strength'] = _compute_deck_strength(battles_df, card_win_rates)
    avg_elixir = battles_df["winner.elixir.average"].mean()
    epsilon_zero = 0.2
    elixir_lower_bound = avg_elixir - epsilon_zero
    elixir_upper_bound = avg_elixir + epsilon_zero
    battles_df["winner.elixir_score"] = battles_df["winner.elixir.average"].apply(lambda x: _elixir_score(x, elixir_lower_bound, elixir_upper_bound))
    battles_df["loser.elixir_score"] = battles_df["loser.elixir.average"].apply(lambda x: _elixir_score(x, elixir_lower_bound, elixir_upper_bound))
    scoring_features = [
    "winner.deck_weighted_strength",
    "winner.avg_card_level",
    "winner.max_card_level",
    "winner.min_card_level",
    "winner.level_variance",
    "winner.elixir_score",
    "winner.synergy_score"
    ]
    scaler = MinMaxScaler()
    battles_df["winner.synergy_score"] = battles_df.apply(_compute_synergy_score, axis=1)
    battles_df[scoring_features] = scaler.fit_transform(battles_df[scoring_features])
    battles_df["winner.deck_final_score"] = (
        0.25 * battles_df["winner.win_rate"] +
        0.25 * battles_df["winner.elixir_score"] +
        0.15 * (battles_df["winner.max_card_level"] + battles_df["winner.min_card_level"]) / 2 +
        0.15 * battles_df["winner.synergy_score"] +
        0.10 * (1 - battles_df["winner.level_variance"]) +
        0.10 * battles_df["winner.avg_card_level"] 
    )
    battles_df = battles_df.round(ROUNDING_PRECISION)
    # Post engineering outliers and noise handling
    battles_df = _post_engineering_outliers_handling(battles_df, deck_total_games)
    return battles_df


def get_numerical_dataset(battles_df):
    """
    Extracts the numerical features from the battle dataset and removes non-informative ID features.

    This function selects only numerical columns from `battles_df` and removes specific 
    identifier columns that do not contribute meaningful information for analysis.

    Parameters:
    - battles_df (pd.DataFrame): The original battle dataset containing both numerical and categorical features.

    Returns:
    - pd.DataFrame: A DataFrame containing only relevant numerical features.
    """
    numerical_columns = battles_df.select_dtypes(include=[np.number]).columns
    id_features_to_remove = ['winner.tag', 'loser.tag', 'gameMode.id']
    df_numeric = battles_df[numerical_columns].copy()
    df_numeric = df_numeric.drop(columns=id_features_to_remove)
    return df_numeric


def get_pca_optimal_components(battles_df):
    """
    Determines the optimal number of PCA components for dimensionality reduction.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset.

    Returns:
    - np.array: Scaled dataset after PCA.
    - int: Optimal number of PCA components.
    """
    df_numeric = get_numerical_dataset(battles_df)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)
    pca = PCA().fit(df_scaled)  # Compute PCA
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n = np.argmax(cumulative_variance >= PCA_VARIENCE_THRESHOLD) + 1
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), cumulative_variance, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=PCA_VARIENCE_THRESHOLD, color='r', linestyle='--')
    plt.show()
    print(f"Best component: {n} with a cumulative_variance value of: {cumulative_variance[n-1]:.4f}")
    return df_scaled, n


def get_t_sne(battles_df, dest_col, preplexity = 30, learning_rate=200, n_iter=1000):
    """
    Computes 3D t-SNE embeddings for visualization.

    Parameters:
    - battles_df (pd.DataFrame): The battle dataset.
    - dest_col (str): Column used for color coding.
    - perplexity (int): t-SNE parameter controlling neighborhood size.
    - learning_rate (int): Step size for optimization.
    - n_iter (int): Number of iterations.

    Returns:
    - pd.DataFrame: DataFrame containing t-SNE embeddings and cluster labels.
    """
    numerical_features = battles_df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical_features)
    tsne_3d = TSNE(n_components=3, perplexity=preplexity, learning_rate=learning_rate, n_iter=n_iter)
    X_embedded_3d = tsne_3d.fit_transform(X_scaled)
    battles_df_tsne_3d = pd.DataFrame(X_embedded_3d, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    battles_df_tsne_3d['Cluster'] = battles_df[dest_col]
    return battles_df_tsne_3d


def plot_t_sne_as_3d_scatter(battles_df_tsne_3d, dest_col):
    """
    Plots a 3D scatter plot of t-SNE embeddings to visualize clusters in the battle dataset.

    Parameters:
    - battles_df_tsne_3d (pd.DataFrame): A DataFrame containing t-SNE embeddings ('TSNE1', 'TSNE2', 'TSNE3') 
      and a 'Cluster' column indicating group assignments.
    - dest_col (str): The name of the column used for color coding in the visualization.

    Returns:
    - None: Displays a 3D scatter plot of t-SNE embeddings.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        battles_df_tsne_3d['TSNE1'], battles_df_tsne_3d['TSNE2'], battles_df_tsne_3d['TSNE3'], 
        c=battles_df_tsne_3d['Cluster'], cmap='viridis', alpha=0.7
    )
    ax.set_title("3D t-SNE Visualization of Battle Data")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(dest_col)
    plt.show()
