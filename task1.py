import kagglehub
import pandas as pd
import os
import glob
from datasets import load_dataset

def standardise_columns(df, source_name):
    df.columns = df.columns.str.lower().str.strip()
    if 'product_name' in df.columns:
        df = df.rename(columns={'product_name': 'item'}) 
    if 'catagory' in df.columns: 
        df = df.rename(columns={'catagory': 'category'})
    if 'item' in df.columns and 'category' in df.columns:
        return df[['item', 'category']]

# KAGGLE DS

path = kagglehub.dataset_download("salahuddinahmedshuvo/grocery-inventory-and-sales-dataset")
csv_files = glob.glob(os.path.join(path, "*.csv"))
    
if csv_files:
    df_kaggle = pd.read_csv(csv_files[0])
    df_kaggle = standardise_columns(df_kaggle, "Kaggle")
    print(f" KAGGLE LOADED")
else:
    df_kaggle = pd.DataFrame()


# AmirMohseni DS 
df_hf1 = pd.read_csv("hf://datasets/AmirMohseni/GroceryList/data.csv")
df_hf1 = standardise_columns(df_hf1, "AmirMohseni")
print(f" AmirMohseni DS  items loaded.")


#HUGGING FACE DS

ds = load_dataset("infinite-dataset-hub/GroceryItemClassification")
df_hf2 = ds['train'].to_pandas()
df_hf2 = standardise_columns(df_hf2, "Infinite-Dataset")
print(f"infinite-dataset-hub items loaded.")


#Processing 
full_df = pd.concat([df_kaggle, df_hf1, df_hf2], ignore_index=True)

# Create a clean search column
full_df['search_name'] = full_df['item'].str.lower().str.strip()

# duplicates
initial_count = len(full_df)
full_df = full_df.drop_duplicates(subset='search_name', keep='last')

def get_grocery_category(item_name):
    """
    Input: Item name (string)
    Output: Category string or formatted message.
    """
    clean_input = item_name.lower().strip()
    match = full_df[full_df['search_name'] == clean_input]
    if not match.empty:
        cat = match.iloc[0]['category']
        return cat
    
    #Partial Match
    partial = full_df[full_df['search_name'].str.contains(clean_input)]
    if not partial.empty:
        best_match = partial.loc[partial['search_name'].str.len().sort_values().index].iloc[0]
        return best_match['category']
        
    return "Unknown Category"

if __name__ == "__main__":
    print("\nTASK 1")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("Enter a grocery item: ")
        
        if user_input.lower() in ['quit', 'exit', 'leave' , 'i am leaving', 'thank you']:
            print("EXITING")
            break
        
        result = get_grocery_category(user_input)
        print(f"--> {result}")
        print("-" * 20)
