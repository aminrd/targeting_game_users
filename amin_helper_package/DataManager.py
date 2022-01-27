import os
import sqlite3
import pyarrow.parquet as pq
import pandas as pd


def load_csv(path, verbose=False):
    """
    Loads a csv file into a pandas dataframe
    :param path: Path to .csv
    :param verbose: Prints the dataframe
    :return: Dataframe
    """
    if verbose:
        print(f"Loading csv file {path}...")
    df = pd.read_csv(path, verbose=verbose)
    return df


def load_sqlite(path, verbose=False):
    """
    Loads a sqlite database into a pandas dataframe
    :param path: Path to .db file
    :param verbose: Prints the dataframe
    :return:
    """
    try:
        if verbose:
            print(f"Loading sqlite file {path}...")

        conn = sqlite3.connect(path)
        df = pd.read_sql_query("SELECT * from devices", conn)
        conn.close()
        return df

    except Exception as e:
        return None


def load_parquet(path, verbose=False):
    """
    Loads a parquet file into a pandas dataframe
    :param path: Path to .parquet file
    :param verbose: prints the dataframe
    :return:
    """
    try:
        if verbose:
            print(f"Loading parquet file {path}...")
        table = pq.read_table(path)
        return table.to_pandas()

    except Exception as e:
        return None


class DataManager:
    def __init__(self, users_path='./data/ka_users.csv',
                 actions_path='./data/ka_actions.parquet',
                 devices_path='./data/ka_devices.db',
                 merge_on='uid_s', verbose=False):
        for path in (users_path, actions_path, devices_path):
            if not isinstance(path, str):
                raise TypeError("Path must be a string")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path}does not exist")

        if verbose:
            print("Loading data from files...")

        self.users = self.read_table(users_path, verbose)
        self.actions = self.read_table(actions_path, verbose)
        self.devices = self.read_table(devices_path, verbose)
        self.merge_on = merge_on

        if verbose:
            print("Data loaded")

    def get_merged_data(self):
        return self.users.merge(self.actions, on=self.merge_on).merge(self.devices, on=self.merge_on)

    def read_table(self, path, verbose=False):
        if path.lower().endswith(".csv"):
            return load_csv(path, verbose)
        elif path.lower().endswith(".db"):
            return load_sqlite(path, verbose)
        elif path.lower().endswith(".parquet"):
            return load_parquet(path, verbose)
        else:
            raise ValueError("File type not supported")
