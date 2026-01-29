import kagglehub
import pandas as pd
import os
import glob
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

def standardise_columns(df, source_name):
    df.columns = df.columns.str.lower().str.strip()
    
    if 'product_name' in df.columns:
        df = df.rename(columns={'product_name': 'item'}) 
    if 'catagory' in df.columns: 
        df = df.rename(columns={'catagory': 'category'})
        
    if 'item' in df.columns and 'category' in df.columns:
        return df[['item', 'category']]
    return pd.DataFrame()



path = kagglehub.dataset_download("salahuddinahmedshuvo/grocery-inventory-and-sales-dataset")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df_kaggle = pd.read_csv(csv_files[0])
df_kaggle = standardise_columns(df_kaggle, "Kaggle")



df_hf1 = pd.read_csv("hf://datasets/AmirMohseni/GroceryList/data.csv")
df_hf1 = standardise_columns(df_hf1, "AmirMohseni")




ds = load_dataset("infinite-dataset-hub/GroceryItemClassification")
df_hf2 = ds['train'].to_pandas()
df_hf2 = standardise_columns(df_hf2, "Infinite-Dataset")



full_df = pd.concat([df_kaggle, df_hf1, df_hf2], ignore_index=True)


full_df = full_df.dropna(subset=['item', 'category'])
full_df['item'] = full_df['item'].astype(str)
full_df['category'] = full_df['category'].astype(str)

print(f"Total Training Data: {len(full_df)} items.")
model = make_pipeline(TfidfVectorizer(), LinearSVC())


X_train = full_df['item']
y_train = full_df['category']

model.fit(X_train, y_train)
print("Finished")

def get_grocery_category(item_name):
    prediction = model.predict([item_name])
    return prediction[0]


if __name__ == "__main__":
    print("\nTASK 1: ")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("Enter a grocery item: ")
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        result = get_grocery_category(user_input)
        print(f"{result}")
        print("---------------------------")
