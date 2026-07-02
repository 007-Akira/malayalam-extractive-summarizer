import pandas as pd
import re
import numpy as np

def segment_and_number(text):
    """Splits Malayalam text into sentences and adds [0], [1] tags."""
    if not isinstance(text, str): return ""
    sentences = re.split(r'\.\s+|\.$', text.strip())
    # Clean up whitespace and remove empty strings
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]
    
    # Format exactly like the gold_dataset_unannotated.csv
    numbered = [f"[{i}] {s}" for i, s in enumerate(sentences)]
    return "\n".join(numbered)

def count_sentences(text):
    """Quick helper to count sentences for filtering."""
    if not isinstance(text, str): return 0
    sentences = re.split(r'\.\s+|\.$', text.strip())
    return len([s for s in sentences if s.strip()])

def main():
    print("1. Loading the massive 90k dataset...")
    # Adjust "data/news_data.csv" if your path is different
    df = pd.read_csv("data/news_data.csv").dropna(subset=['Text', 'Summary'])
    
    print("\n2. Scanning for the 'Goldilocks' articles (6 to 15 sentences)...")
    # Apply sentence counting
    df['Sentence_Count'] = df['Text'].apply(count_sentences)
    
    # Filter for the highest quality annotation targets
    best_articles_df = df[(df['Sentence_Count'] >= 6) & (df['Sentence_Count'] <= 15)].copy()
    print(f"   Found {len(best_articles_df)} perfectly sized articles out of {len(df)}.")
    
    print("\n3. Extracting 700 random articles from the best pool...")
    # Grab 700 random articles from this high-quality pool
    platinum_df = best_articles_df.sample(n=700, random_state=42).copy()
    
    print("\n4. Formatting to match the Gold Dataset structure...")
    # Create the numbered source text
    platinum_df['Numbered_Source_Text'] = platinum_df['Text'].apply(segment_and_number)
    
    # Map the columns to match your uploaded CSV format
    # Generate an ID if one doesn't exist, or use the dataframe index
    if 'id' in platinum_df.columns:
        platinum_df['Article_ID'] = platinum_df['id']
    else:
        platinum_df['Article_ID'] = platinum_df.index

    platinum_df['Original_Abstractive_Summary'] = platinum_df['Summary']
    
    # Create the blank columns for your team
    platinum_df['Your_Extractive_Labels'] = ""
    platinum_df['Annotator_Name'] = "" # So you, Goutham, and Sredha can claim rows
    
    # Reorder columns perfectly
    final_df = platinum_df[[
        'Article_ID', 
        'Numbered_Source_Text', 
        'Original_Abstractive_Summary', 
        'Your_Extractive_Labels',
        'Annotator_Name'
    ]]
    
    # Save it!
    output_path = "data/platinum_batch_700.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n==========================================")
    print(f"✅ Success! Saved perfectly formatted dataset to:")
    print(f"   {output_path}")
    print(f"==========================================")

if __name__ == "__main__":
    main()