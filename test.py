import amin_helper_package as helper

data_manager = helper.DataManager(users_path='./data/ka_users.csv',
                                  actions_path='./data/ka_actions.parquet',
                                  devices_path='./data/ka_devices.db',
                                  merge_on='uid_s', verbose=True)

df_merged = data_manager.get_merged_data()

