import kagglehub
import pandas as pd
import numpy as np
import os
import glob
import sys
import random

def load_kaggle_data():
    """ Load Source 1: Kaggle """
    

    path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files: return pd.DataFrame()

    df = pd.read_csv(csv_files[0])
    df['Transaction_ID'] = 'K_' + df['Member_number'].astype(str) + '_' + df['Date'].astype(str)
    df['item'] = df['itemDescription'].astype(str).str.lower().str.strip()
    print("LOADED KAGGLE DB")
    return df[['Transaction_ID', 'item']]


def load_uploaded_csv():
    if not os.path.exists('groceries - groceries.csv'):
        return pd.DataFrame()


    df = pd.read_csv('groceries - groceries.csv')
    df['Transaction_ID'] = 'U_' + df.index.astype(str)
    item_cols = [c for c in df.columns if c.startswith('Item') and c != 'Item(s)']
    melted = df.melt(id_vars=['Transaction_ID'], value_vars=item_cols, value_name='item')
    melted = melted.dropna(subset=['item'])
    melted['item'] = melted['item'].astype(str).str.lower().str.strip()
    print("Loaded Github CSV")
    return melted[['Transaction_ID', 'item']]


def build_model():
    df_kaggle = load_kaggle_data()
    df_upload = load_uploaded_csv()
    
    full_data = pd.concat([df_kaggle, df_upload], ignore_index=True)

        
    initial_item_count = full_data['item'].nunique()

    
    item_counts = full_data['item'].value_counts()
    items_to_keep = item_counts[item_counts > 7].index 
    full_data = full_data[full_data['item'].isin(items_to_keep)]
    
    removed_count = initial_item_count - full_data['item'].nunique()
    
    basket_matrix = pd.crosstab(full_data['Transaction_ID'], full_data['item'])
    basket_matrix = (basket_matrix > 0).astype(int)
    co_occurrence = basket_matrix.T.dot(basket_matrix)
    np.fill_diagonal(co_occurrence.values, 0)
    
    item_frequencies = basket_matrix.sum(axis=0)
    

    scoresMatrix = co_occurrence.div(item_frequencies, axis=1)
    
    print("Model Built Successfully.")
    return scoresMatrix, item_frequencies

def smart_map_input(user_item, item_list, popularity_series):
    user_item = user_item.strip().lower()
    if user_item in item_list: return user_item

    matches = [i for i in item_list if user_item in i]
    if not matches: return None
    
    best_match = max(matches, key=lambda x: popularity_series.get(x, 0))
    return best_match

def recommend_item(basket_items, model, popularity):

    mapped_basket = []
    for item in basket_items:
        match = smart_map_input(item, model.index, popularity)
        if match:
            mapped_basket.append(match)



    scores = model.loc[mapped_basket].sum()
    scores = scores.drop(labels=mapped_basket, errors='ignore')

    top_candidates = scores.nlargest(3)
    chosen_item = random.choice(top_candidates.index)
    chosen_score = top_candidates[chosen_item]
    
    return chosen_item, chosen_score




model_matrix, item_pop = build_model()

print("\n Task 2: " )
while True:
    user_input = input("Enter Basket (Separete with ,) or 'quit': ")
    
    if user_input.lower() in ['quit', 'exit']:
        break
    
    basket = [x.strip() for x in user_input.split(',') if x.strip()]
    if not basket: continue
    
    rec, score = recommend_item(basket, model_matrix, item_pop)
    
    if rec:
        print(f"\nRecommendation: {rec.upper()}")
    else:
        print("\nNot in DS")
