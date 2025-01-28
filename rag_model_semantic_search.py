# -*- coding: utf-8 -*-
"""rag_model_semantic_search.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oQDNQZTdMFMlP8aXQaW18Owxkc-fkTpf
"""

#pip install pandas numpy transformers pinecone-client torch scikit-learn

"""# **Upload to Pinecone**"""

if __name__ == "__main__":

  #pip install pandas numpy transformers pinecone-client torch scikit-learn

  import pandas as pd
  import numpy as np
  from transformers import AutoTokenizer, AutoModel
  import pinecone
  import torch

  from google.colab import files
  uploaded = files.upload()

  data = pd.read_csv("Trade_Data_Raw.csv")
  data.head(1)

  data = data.dropna()

  data['json'] = data.apply(
      lambda row:{
          "Time" : row['Time'],
          "Month" : row['Month'],
          "Units" : row['Units'],
          "Base_Price" : row['Base_Price'],
          "Non_Promoted_Price" : row['Non_Promoted_Price'],
          "Customer" : row['Customer'],
          "Item" : row['Item'],

          "Week_Type" : row['Week_Type'],
          "Promo_Type" : row['Promo_Type'],
          "Merch" : row['Merch'],
          "List_Price" : row['List_Price'],

          "COGS_Unit" : row['COGS_Unit'],
          "Retailer_Desired_Margin_%": row['Retailer_Desired_Margin_%'],
          "Retailer_No_Promo_$_Margin" : row['Retailer_No_Promo_$_Margin'],

          "EDLP_Trade_$_Unit" : row['EDLP_Trade_$_Unit'],
          "$_Margin_this_week" : row['$_Margin_this_week'],
          "Var_Trade_$_Unit" : row['Var_Trade_$_Unit'],
          "$_Total_Trade_Unit" : row['$_Total_Trade_Unit'],

          "%_Trade_Rate" : row['%_Trade_Rate'],
          "Manufacturer_$_Profit_Unit" : row['Manufacturer_$_Profit_Unit'],
          "Manufacturer_%_Proft_Unit" : row['Manufacturer_%_Proft_Unit']
      },
      axis=1
  )

  data['embedding_input'] = data['json'].apply(str)

  model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)

  data.head(1)

  def generate_embedding(text):
    inputs = tokenizer(text, return_tensors = "pt", truncation = True, max_length = 512, padding = True)
    with torch.no_grad():
      outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


  data['embedding'] = data['embedding_input'].apply(generate_embedding)

  from pinecone import Pinecone, ServerlessSpec

  # Initialize Pinecone with your API key
  pc = Pinecone(api_key="your_api_key")

  index_name = "trade-data-embeddings"

  pc.create_index(
      name = index_name,
      dimension = 768,
      metric = "cosine",
      spec = ServerlessSpec(
          cloud = 'aws',
          region = 'us-east-1'
      )
  )

  index = pc.Index(index_name)

  import json  # Add this import to handle JSON serialization

  for idx, row in data.iterrows():
      # Serialize the JSON metadata to a string
      metadata = {"json": json.dumps(row['json'])}

      # Upsert the embedding with serialized metadata
      index.upsert([
          (str(idx), row['embedding'], metadata)
      ])

"""# **Generate Predictions**"""

from pinecone import Pinecone, ServerlessSpec

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import torch
import pinecone
from sklearn.metrics.pairwise import cosine_similarity

from pinecone import Pinecone

# Initialize Pinecone with API key
pc = Pinecone(
    api_key="your_api_key",
)

# Connect to the existing index
index_name = "trade-data-embeddings"
index = pc.Index(index_name)

# Retrieve index stats
stats = index.describe_index_stats()
print(stats)

"""# **Method 1 - Only use the embedding model**"""

# Load the embedding model
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)



from sentence_transformers import SentenceTransformer

# Load model using Sentence-Transformers
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

def generate_query_embedding(query):
    # Generate the embedding directly
    embedding = embedding_model.encode(query)
    return embedding

import json

def retrieve_from_pinecone(query, top_k=5):
    # Generate the query embedding as a NumPy array
    query_embedding = generate_query_embedding(query)

    # Convert NumPy array to a Python list
    query_embedding_list = query_embedding.tolist()

    # Query Pinecone
    results = index.query(
        vector=query_embedding_list,  # Pass the embedding as a list
        top_k=top_k,
        include_metadata=True
    )

    # Extract metadata from the results
    retrieved_context = [json.loads(res["metadata"]["json"]) for res in results["matches"]]
    return retrieved_context

query = "Give me top 5 promotions for item Item_177 at customer Discount"

retrieve_from_pinecone(query, top_k=5)

"""# **Use the embedding model along with Generative Model**"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Use your Hugging Face token
hf_token = "your_token"

generative_models = {
    "BERT": "bert-base-uncased",
    "GPT-2": "gpt2"
}

loaded_models = {}

for model_name, model_path in generative_models.items():
  print(f"Loading {model_name}")
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  # Add padding token if missing
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

  model = AutoModelForCausalLM.from_pretrained(model_path)
  loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}



def generate_response_with_model(query, context, model_name):
  tokenizer = loaded_models[model_name]["tokenizer"]
  model = loaded_models[model_name]["model"]

  context_text = "\n".join([json.dumps(c) for c in context])
  input_text = f"Query: {query}\nContext: {context_text}"

  #Tokenize Inputs
  inputs = tokenizer(input_text, return_tensors = 'pt', truncation = True, max_length = 512, padding = True)
  outputs = model.generate(**inputs, max_length = 746)
  response = tokenizer.decode(outputs[0], skip_special_tokens = True)
  return response
