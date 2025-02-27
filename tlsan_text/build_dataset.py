import argparse
import random
import pickle
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

max_length = 90
random.seed(1234)


# === Load your data ===

parser = argparse.ArgumentParser(description='Load and convert pickle to CSV.')
parser.add_argument('--dataset', type=str, help='Name of the dataset to process')

# Parse the arguments
args = parser.parse_args()
dataset=args.dataset


with open('new_data/'+dataset+'.pkl', 'rb') as f:
    reviews_df, meta_df = pickle.load(f)
    item_cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

# Load review embeddings CSV file.
# (Make sure the CSV file contains columns "reviewEmbedding" and "asin".)
review_emb_df = pd.read_csv('new_data/'+dataset+'_embedding.csv')
# If the CSV does not have an "asin" column, you might assign it from reviews_df.
# For example, if reviews_df is a DataFrame, you might do:
# review_emb_df['asin'] = reviews_df['asin'].tolist()


# === Helper function for time embeddings ===

# Generate current time positions (gaps)
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


def proc_time_emb(hist_t, cur_t):
    # For each historical time, compute a reciprocal sum over how many gaps the time difference spans.
    hist_t = [cur_t - i + 1 for i in hist_t]
    hist_t = [1 / np.sum(np.array(i) >= gap) for i in hist_t]
    return hist_t


# === Build train and test sets as lists of dictionaries ===

train_set = []
test_set = []

# Iterate over each reviewer.
for reviewerID, hist in tqdm(reviews_df.groupby('reviewerID')):
    pos_list = hist['asin'].tolist()
    tim_list = hist['unixReviewTime'].tolist()

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg() for _ in range(len(pos_list))]

    length = len(pos_list)
    valid_length = min(length, max_length)
    i = 0
    tim_list_session = list(set(tim_list))
    tim_list_session.sort()
    pre_session = []
    pre_time = []
    pre_cates = []

    for t in tim_list_session:
        count = tim_list.count(t)
        new_session = pos_list[i:i + count]
        new_time = tim_list[i:i + count]
        # Get categories for the items in new_session.
        new_cates = [meta_df[meta_df['asin'] == item]['categories'].tolist()[0] for item in new_session]

        if t == tim_list_session[0]:
            pre_session.extend(new_session)
            pre_time.extend(new_time)
            pre_cates.extend(new_cates)
        else:
            now_cate = pd.value_counts(pre_cates).index[0]
            if i + count < valid_length - 1:
                pre_time_emb = proc_time_emb(pre_time, tim_list[i])
                pre_session_copy = copy.deepcopy(pre_session)

                # --- Positive training sample ---
                target_item_pos = pos_list[i + count]
                rows = review_emb_df[
                    (review_emb_df['asin'] == target_item_pos) & (review_emb_df['reviewerID'] == reviewerID)]
                if rows.empty:
                    continue
                else:
                    embeed_str=np.fromstring(rows.iloc[0]['reviewEmbedding'][1:-1], sep=' ')

                train_set.append({
                    "reviewerID": reviewerID,
                    "history": pre_session_copy,
                    "session": new_session,
                    "time_embedding": pre_time_emb,
                    "target_item": target_item_pos,
                    "label": 1,
                    "current_category": now_cate,
                    "review_embedding": embeed_str
                })

                # --- Negative training sample ---
                target_item_neg = neg_list[i + count]
                rows = review_emb_df[(review_emb_df['asin'] == target_item_neg)]
                if rows.empty:
                    continue
                else:
                    embeed_str=np.fromstring(rows.iloc[0]['reviewEmbedding'][1:-1], sep=' ')

                train_set.append({
                    "reviewerID": reviewerID,
                    "history": pre_session_copy,
                    "session": new_session,
                    "time_embedding": pre_time_emb,
                    "target_item": target_item_neg,
                    "label": 0,
                    "current_category": now_cate,
                    "review_embedding": embeed_str
                })

                pre_session.extend(new_session)
                pre_time.extend(new_time)
                pre_cates.extend(new_cates)
            else:
                # --- Test record ---
                pos_item = pos_list[i]
                if count > 1:
                    pos_item = random.choice(new_session)
                    new_session.remove(pos_item)
                neg_index = pos_list.index(pos_item)
                pre_time_emb = proc_time_emb(pre_time, t)

                rows = review_emb_df[(review_emb_df['asin'] == pos_item) & (review_emb_df['reviewerID'] == reviewerID)]
                if rows.empty:
                    continue
                else:
                    embeed_str=np.fromstring(rows.iloc[0]['reviewEmbedding'][1:-1], sep=' ')

                test_set.append({
                    "reviewerID": reviewerID,
                    "history": pre_session,
                    "session": new_session,
                    "time_embedding": pre_time_emb,
                    "target_items": {"positive": pos_item, "negative": neg_list[neg_index]},
                    "current_category": now_cate,
                    "review_embedding": embeed_str
                })
                break  # finish processing this reviewer
        i += count

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

with open('Prepared_Dataset/'+dataset+'.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(item_cate_list, f, pickle.HIGHEST_PROTOCOL)
