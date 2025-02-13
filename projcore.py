import os
import kaggle

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

def feature_engineering():
    pass

