import pandas as pd
import os

def main():
    # Updated to match your actual file name
    dataset_path = "data/news_data.csv" 
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        print("Please create a 'data' folder and place your dataset inside it.")
        return

    print(f"1. Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print("\n2. Initial Dataset Statistics:")
    print(f"   Total Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Updated to match the exact column names in your CSV
    print("\n3. Cleaning Missing Values...")
    df = df.dropna(subset=['Text', 'Summary'])
    print(f"   Rows remaining after dropping nulls: {len(df)}")
    
    print("\n4. Dropping Duplicates...")
    df = df.drop_duplicates(subset=['Text'])
    print(f"   Final Cleaned Row Count: {len(df)}")
    
    print("\n5. Previewing first clean record:")
    print("-" * 50)
    print("ARTICLE PREVIEW:")
    print(str(df.iloc[0]['Text'])[:200] + "...")
    print("\nSUMMARY PREVIEW:")
    print(str(df.iloc[0]['Summary'])[:200] + "...")
    print("-" * 50)

if __name__ == "__main__":
    main()