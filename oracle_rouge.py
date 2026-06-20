import pandas as pd
import re
from rouge_score import rouge_scorer
from tqdm import tqdm
import os

def segment_malayalam_text(text):
    if not isinstance(text, str): return []
    sentences = re.split(r'\.\s+|\.$', text.strip())
    return [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]

def main():
    print("1. Loading the FULL 90k dataset...")
    # Read the full dataset, dropping any broken rows
    df = pd.read_csv("data/news_data.csv").dropna(subset=['Text', 'Summary'])
    print(f"   Found {len(df)} valid articles. Buckle up.")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    training_data = []
    
    print("\n2. Generating Binary Labels via Oracle ROUGE...")
    
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = segment_malayalam_text(row['Text'])
        target_summary = str(row['Summary'])
        
        total_sentences = len(sentences)
        if total_sentences < 3: continue
            
        scores = []
        for i, sentence in enumerate(sentences):
            score = scorer.score(target_summary, sentence)['rougeL'].fmeasure
            scores.append((i, score, sentence))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in scores[:3]]
        
        for i, _, sentence in scores:
            label = 1 if i in top_indices else 0
            training_data.append({
                'Sentence': sentence,
                'Label': label,
                'Sentence_Index': i,
                'Total_Sentences': total_sentences
            })
            
    # Save as a compressed gzip file to bypass GitHub's 100MB limit
    print("\n3. Compressing and saving massive dataset...")
    final_train_df = pd.DataFrame(training_data)
    final_train_df.to_csv("data/training_data.csv.gz", compression="gzip", index=False)
    
    print("\n==========================================")
    print(f"✅ Success! Created {len(final_train_df)} labeled training sentences.")
    print("Saved securely to data/training_data.csv.gz")
    print("==========================================")

if __name__ == "__main__":
    main()