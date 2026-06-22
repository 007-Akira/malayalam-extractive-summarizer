import pandas as pd
import re
from rouge_score import rouge_scorer
from tqdm import tqdm
import os

def segment_and_number(text):
    """Splits Malayalam text into sentences and adds [0], [1] tags."""
    if not isinstance(text, str): return "", []
    sentences = re.split(r'\.\s+|\.$', text.strip())
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]
    numbered = [f"[{i}] {s}" for i, s in enumerate(sentences)]
    return "\n".join(numbered), sentences

def main():
    print("="*50)
    print("💎 GENERATING DIAMOND TEST BATCH (1,000 ARTICLES)")
    print("="*50)
    
    # 1. Load the massive dataset
    print("1. Loading original news_data.csv...")
    try:
        df = pd.read_csv("data/news_data.csv").dropna(subset=['Text', 'Summary'])
    except FileNotFoundError:
        print("Error: data/news_data.csv not found!")
        return

    # 2. Exclude the Platinum Batch to ensure NO data leakage!
    try:
        platinum_df = pd.read_csv("data/platinum_batch_700.csv")
        # Ensure we have consistent ID matching
        df['temp_id'] = df['id'].astype(str) if 'id' in df.columns else df.index.astype(str)
        plat_ids = set(platinum_df['Article_ID'].astype(str))
        
        original_len = len(df)
        df = df[~df['temp_id'].isin(plat_ids)]
        print(f"   Excluded {original_len - len(df)} articles already used in the Platinum batch.")
    except FileNotFoundError:
        print("   (Platinum batch not found, skipping exclusion step.)")

    # 3. Filter for Goldilocks size (6 to 15 sentences)
    print("2. Filtering for optimal article lengths...")
    df['Sentence_Count'] = df['Text'].apply(lambda x: len(re.split(r'\.\s+|\.$', str(x).strip())))
    best_articles_df = df[(df['Sentence_Count'] >= 6) & (df['Sentence_Count'] <= 15)].copy()
    
    # 4. Sample 1000 entirely new articles
    print("3. Sampling 1,000 new articles...")
    # We use a different random_state (99) to ensure a completely different slice of data
    diamond_df = best_articles_df.sample(n=1000, random_state=99).copy()
    
    if 'id' in diamond_df.columns:
        diamond_df['Article_ID'] = diamond_df['id']
    else:
        diamond_df['Article_ID'] = diamond_df.index
        
    diamond_df['Original_Abstractive_Summary'] = diamond_df['Summary']

    # 5. Run the Oracle Auto-Annotator
    print("4. Firing up the ROUGE Oracle to auto-annotate the 1,000 articles...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    numbered_texts = []
    auto_labels = []
    
    for idx, row in tqdm(diamond_df.iterrows(), total=len(diamond_df)):
        numbered_text, sentences = segment_and_number(row['Text'])
        summary = str(row['Original_Abstractive_Summary'])
        
        numbered_texts.append(numbered_text)
        
        if not sentences or summary == "nan":
            auto_labels.append("")
            continue
            
        # Score every sentence
        scores = []
        for i, sentence in enumerate(sentences):
            score = scorer.score(summary, sentence)['rougeL'].fmeasure
            scores.append((i, score))
            
        # Sort and pick top 3
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(3, len(sentences))
        best_indices = sorted([x[0] for x in scores[:top_k]])
        
        auto_labels.append(", ".join(map(str, best_indices)))

    diamond_df['Numbered_Source_Text'] = numbered_texts
    diamond_df['Your_Extractive_Labels'] = auto_labels
    
    # Clean up and save
    final_df = diamond_df[['Article_ID', 'Numbered_Source_Text', 'Original_Abstractive_Summary', 'Your_Extractive_Labels']]
    output_path = "data/diamond_batch_1000_ANNOTATED.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Success! Diamond Batch created and annotated.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()