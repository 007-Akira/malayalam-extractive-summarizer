import pandas as pd
import re
import os

def segment_and_number(text):
    """Splits Malayalam text into sentences and adds [0], [1] tags."""
    if not isinstance(text, str): return ""
    sentences = re.split(r'\.\s+|\.$', text.strip())
    # Clean up whitespace and remove empty strings
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]
    
    # Format identically to the screenshot
    numbered = [f"[{i}] {s}" for i, s in enumerate(sentences)]
    return "\n".join(numbered)

def count_sentences(text):
    """Quick helper to count sentences for filtering."""
    if not isinstance(text, str): return 0
    sentences = re.split(r'\.\s+|\.$', text.strip())
    return len([s for s in sentences if s.strip()])

def main():
    print("="*50)
    print("🧑‍🤝‍🧑 GENERATING TEAM ANNOTATION BATCH (50 ARTICLES)")
    print("="*50)
    
    print("1. Loading the massive dataset (requiring both text and summaries)...")
    try:
        # We need both Text and Summary now!
        df = pd.read_csv("data/news_data.csv").dropna(subset=['Text', 'Summary'])
    except FileNotFoundError:
        print("Error: data/news_data.csv not found!")
        return
    
    print("2. Scanning for 'Goldilocks' articles (6 to 15 sentences)...")
    df['Sentence_Count'] = df['Text'].apply(count_sentences)
    best_articles_df = df[(df['Sentence_Count'] >= 6) & (df['Sentence_Count'] <= 15)].copy()
    
    # Exclude articles already used in Platinum and Diamond if possible
    # We will use random_state=101 to grab a completely fresh slice of data
    print("3. Extracting 50 entirely new articles...")
    human_df = best_articles_df.sample(n=50, random_state=101).copy()
    
    print("4. Formatting source text and setting up team columns...")
    human_df['Numbered_Source_Text'] = human_df['Text'].apply(segment_and_number)
    
    if 'id' in human_df.columns:
        human_df['Article_ID'] = human_df['id']
    else:
        human_df['Article_ID'] = human_df.index
    
    # Add the Abstractive Summary back in!
    human_df['Original_Abstractive_Summary'] = human_df['Summary']
    
    # Create the dedicated blank columns for each annotator
    human_df['Godly_Labels'] = ""
    human_df['Bastin_Labels'] = ""
    human_df['Adithya_Labels'] = ""
    
    # Reorder columns to include the Abstractive Summary
    final_df = human_df[[
        'Article_ID', 
        'Numbered_Source_Text', 
        'Original_Abstractive_Summary',
        'Godly_Labels',
        'Bastin_Labels',
        'Adithya_Labels'
    ]]
    
    # Save the file
    os.makedirs("data", exist_ok=True)
    output_path = "data/human_annotation_batch_50.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Success! Dataset created.")
    print(f"Saved to: {output_path}")
    print("Distribute this CSV to Godly and Bastin. Have everyone fill in their columns like '0, 1, 3' without looking at each other's answers!")

if __name__ == "__main__":
    main()