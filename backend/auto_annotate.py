import pandas as pd
import re
from rouge_score import rouge_scorer
from tqdm import tqdm

def extract_clean_sentences(numbered_text):
    """Strips the '[0]' tags so the algorithm can evaluate the raw text."""
    if not isinstance(numbered_text, str): return []
    lines = numbered_text.split('\n')
    sentences = []
    for line in lines:
        # Regex to remove the "[0] " prefix
        clean_text = re.sub(r'^\[\d+\]\s*', '', line).strip()
        if clean_text:
            sentences.append(clean_text)
    return sentences

def main():
    # Note: Change this filename to match whichever file you want to auto-annotate
    # (e.g., "gold_dataset_unannotated.csv" or "data/platinum_batch_700.csv")
    input_file = "data/platinum_batch_700.csv"
    output_file = "data/platinum_batch_700_ANNOTATED.csv"
    
    print(f"1. Loading dataset: {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please check the path.")
        return

    print("2. Firing up the Oracle Auto-Annotator...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    auto_labels = []
    
    # Iterate through every article in the CSV
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sentences = extract_clean_sentences(row['Numbered_Source_Text'])
        summary = str(row['Original_Abstractive_Summary'])
        
        # If the text is broken or missing, leave the label blank
        if not sentences or summary == "nan":
            auto_labels.append("")
            continue
            
        # Score every sentence against the summary
        scores = []
        for i, sentence in enumerate(sentences):
            score = scorer.score(summary, sentence)['rougeL'].fmeasure
            scores.append((i, score))
            
        # Sort sentences by highest ROUGE overlap score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Pick the Top 3 best matching sentences
        top_k = min(3, len(sentences))
        best_indices = [x[0] for x in scores[:top_k]]
        
        # Sort the numbers sequentially (e.g., [0, 2, 4])
        best_indices.sort()
        
        # Convert list to a string like "0, 2, 4"
        label_string = ", ".join(map(str, best_indices))
        auto_labels.append(label_string)
        
    # Write the AI's choices directly into your blank column
    df['Your_Extractive_Labels'] = auto_labels
    
    # Save the completed file
    print("\n3. Saving annotated dataset...")
    df.to_csv(output_file, index=False)
    
    print(f"==========================================")
    print(f"✅ Success! Auto-annotated {len(df)} articles.")
    print(f"Saved to: {output_file}")
    print(f"==========================================")

if __name__ == "__main__":
    main()