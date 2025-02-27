import pandas as pd
import pickle

def update_and_save_data(dataset):
    # Path to the CSV and the pickle files
    csv_path = "new_data/" +dataset + "_embed.csv"
    pickle_path = "new_data/" + dataset + ".pkl"

    # Load the existing data from the pickle file
    with open(pickle_path, 'rb') as f:
        reviews_df, meta_df = pickle.load(f)  # Load the first tuple
        item_cate_list = pickle.load(f)       # Load the second object
        counts = pickle.load(f)               # Load the second tuple

    # Load new data from the CSV
    new_data = pd.read_csv(csv_path)

    # Append new data to reviews_df as 'embed' column
    reviews_df = pd.concat([reviews_df, new_data], axis=1)

    # Save the updated dataframes and lists back to the pickle file
    with open("new_data/" + dataset + "_final_embed.pkl", 'wb') as f:
        pickle.dump((reviews_df, meta_df), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(item_cate_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(counts, f, pickle.HIGHEST_PROTOCOL)

update_and_save_data('Books')
print(44)
