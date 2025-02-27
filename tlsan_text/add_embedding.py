import argparse
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import tqdm

parser = argparse.ArgumentParser(description='Load and convert pickle to CSV.')
parser.add_argument('--dataset', type=str, help='Name of the dataset to process')

# Parse the arguments
args = parser.parse_args()
dataset=args.dataset

Data = 'new_data/'
reviews_df = pd.read_pickle(Data + dataset+'.pkl')

def embed_reviews(df,device):
  # This function assumes there is a 'reviewText' column in the dataframe
  review_embedding = []
  model = SentenceTransformer('shafin/distilbert-similarity-b32')
  model.to(device)

  for review in tqdm.tqdm(df[0]['reviewText']):

    embedding = model.encode(review)

    # Append the result to the list
    review_embedding.append(embedding)


  # Replace the 'reviewText' column with 'reviewEmbedding'
  df[0]['reviewEmbedding'] = review_embedding

  return df



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reviews_df = embed_reviews(reviews_df,device)

## save the reviewEmbedding column as a csv file
reviews_df[0].to_csv(Data + dataset+'_embedding.csv',index=False)