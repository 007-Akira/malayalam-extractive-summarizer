import pandas as pd
import re
from summarize import summarize_article

def main():
    print("="*50)
    print("🚀 TESTING CROSS-DOMAIN GENERALIZATION (WIKIPEDIA)")
    print("="*50)
    
    # Load the Wikipedia dataset you just generated
    try:
        df = pd.read_csv("data/wikipedia_ood_batch.csv")
    except FileNotFoundError:
        print("Error: data/wikipedia_ood_batch.csv not found!")
        print("Please run 'python generate_wikipedia_test.py' first.")
        return

    # Map the Article_IDs directly to their true names
    topic_map = {
        "WIKI_0": "കേരളം (Kerala)",
        "WIKI_1": "ഇസ്രോ (ISRO)",
        "WIKI_2": "നിർമ്മിത ബുദ്ധി (AI)",
        "WIKI_3": "മോഹൻലാൽ (Mohanlal)",
        "WIKI_4": "ഓണം (Onam)"
    }

    for idx, row in df.iterrows():
        article_id = str(row['Article_ID'])
        topic_name = topic_map.get(article_id, article_id)
        
        # The text is numbered (e.g. "[0] sentence"). The generator stripped the periods!
        # Let's clean the tags and add the periods back so the AI can segment it properly.
        clean_lines = []
        for line in str(row['Numbered_Source_Text']).split('\n'):
            clean_text = re.sub(r'\[\d+\]\s*', '', line).strip()
            if clean_text:
                clean_lines.append(clean_text + ".")
        raw_text = " ".join(clean_lines)
        
        print(f"\n📚 Summarizing Topic: {topic_name}")
        
        # Test the AI with Dynamic MMR!
        try:
            summary, _ = summarize_article(raw_text, k=3, diversity="auto")
            print(f"✨ AI Summary:\n{summary}")
        except Exception as e:
            print(f"Error summarizing {topic_name}: {e}")

if __name__ == "__main__":
    main()