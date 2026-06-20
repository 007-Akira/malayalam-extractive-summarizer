import pandas as pd
import re

def segment_malayalam_text(text):
    """Splits text safely and cleans whitespace."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r'\.\s+|\.$', text.strip())
    return [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]

def main():
    dataset_path = "data/news_data.csv"
    output_path = "data/gold_dataset_unannotated.csv"
    
    print("1. Loading the 90k dataset...")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['Text', 'Summary'])
    
    print("2. Randomly sampling 100 articles...")
    # random_state=42 ensures we get the exact same 100 articles every time we run this
    gold_df = df.sample(n=100, random_state=42).copy()
    
    print("3. Pre-segmenting sentences for human annotation...")
    formatted_rows = []
    
    for index, row in gold_df.iterrows():
        sentences = segment_malayalam_text(row['Text'])
        
        # Create a readable numbered list of sentences for the spreadsheet
        numbered_text = "\n".join([f"[{i}] {s}" for i, s in enumerate(sentences)])
        
        formatted_rows.append({
            'Article_ID': index,
            'Numbered_Source_Text': numbered_text,
            'Original_Abstractive_Summary': row['Summary'],
            'Your_Extractive_Labels': '' # <--- This is where you will type e.g., "0, 3"
        })
        
    final_gold_df = pd.DataFrame(formatted_rows)
    final_gold_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Success! Saved 100 pre-segmented articles to: {output_path}")
    print("Open this CSV file in Excel, Google Sheets, or VS Code.")

if __name__ == "__main__":
    main()