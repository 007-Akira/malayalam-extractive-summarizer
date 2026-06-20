import pandas as pd
import re
from rouge_score import rouge_scorer
from tqdm import tqdm

def segment_malayalam_text(text):
    if not isinstance(text, str): return []
    sentences = re.split(r'\.\s+|\.$', text.strip())
    return [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]

def main():
    print("1. Loading the 90k dataset...")
    df = pd.read_csv("data/news_data.csv").dropna(subset=['Text', 'Summary'])
    
    # We only need 5,000 articles to train a classification head
    print("2. Sampling 5,000 articles for the training set...")
    train_df = df.sample(n=5000, random_state=101).copy()
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    training_data = []
    
    print("3. Generating Binary Labels via Oracle ROUGE...")
    # tqdm gives us a nice progress bar
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        sentences = segment_malayalam_text(row['Text'])
        target_summary = str(row['Summary'])
        
        if len(sentences) < 3: continue
            
        scores = []
        for i, sentence in enumerate(sentences):
            # Compare each sentence to the actual human summary
            score = scorer.score(target_summary, sentence)['rougeL'].fmeasure
            scores.append((i, score, sentence))
            
        # Sort by highest ROUGE score and pick the top 3 sentences
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in scores[:3]]
        
        # Assign 1 if it's in the top 3, 0 otherwise
        for i, _, sentence in scores:
            label = 1 if i in top_indices else 0
            training_data.append({
                'sentence': sentence,
                'label': label
            })
            
    # Save our massive new training dataset
    final_train_df = pd.DataFrame(training_data)
    final_train_df.to_csv("data/training_data.csv", index=False)
    
    print("\n==========================================")
    print(f"✅ Success! Created {len(final_train_df)} labeled training sentences.")
    print("Saved to data/training_data.csv")
    print("==========================================")

if __name__ == "__main__":
    main()