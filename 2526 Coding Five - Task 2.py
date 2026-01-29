import kagglehub
import pandas as pd
import os
import glob
import sys

def load_data():
    path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.read_csv(csv_files[0])
        
    df['Transaction_ID'] = df['Member_number'].astype(str) + '_' + df['Date'].astype(str)
        
    df['item'] = df['itemDescription'].astype(str).str.lower().str.strip()
        
    print(f"-> Loaded Amount:{len(df)}")
        
    return df[['Transaction_ID', 'item']]

def get_basic_recommendation(user_input, df):
    user_item = user_input.lower().strip()


    matching_txns = df[df['item'] == user_item]['Transaction_ID'].unique()
    if len(matching_txns) == 0:
        return None, 0

    rows = df[df['Transaction_ID'].isin(matching_txns)]
    

    other_items = rows[co_occurring_rows['item'] != user_item]
    
    if other_items.empty:
        return None, 0
    

    item_counts = other_items['item'].value_counts()
    

    recommendation = item_counts.index[0]      
    count = item_counts.iloc[0]                
    total_baskets = len(matching_txns)         
    
    confidence = (count / total_baskets) * 100
    
    return recommendation, confidence

if __name__ == "__main__":
    df = load_data()
    
    if not df.empty:
        print("\nTASK 2:")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\nEnter: ")
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            rec, conf = get_basic_recommendation(user_input, df)
            
            if rec and rec != "Unknown Item":
                print(f"Recommendation: {rec.upper()}")
                print(f"   (Confidence: {conf:.1f})")
            elif rec == "Unknown Item":
                print("I dont know what this is.")
            else:
                print("No Pattern.")
