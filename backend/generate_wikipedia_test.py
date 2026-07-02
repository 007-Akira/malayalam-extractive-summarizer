import pandas as pd
import requests
import re

def fetch_malayalam_wikipedia(title):
    """Fetches plain text from a Malayalam Wikipedia article."""
    url = f"https://ml.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True
    }
    
    # Adding a User-Agent header is often required by Wikipedia's API
    headers = {
        "User-Agent": "MalayalamSummarizerBot/1.0 (https://github.com/007-Akira/malayalam-extractive-summarizer)"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status() # Check for HTTP errors (like 404 or 500)
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        for page_id in pages:
            # Check if the page actually exists
            if page_id == "-1":
                print(f"      [Warning] Page '{title}' not found on Wikipedia.")
                return ""
            return pages[page_id].get('extract', '')
    except requests.exceptions.RequestException as e:
        print(f"      [Error] Network request failed for '{title}': {e}")
    except ValueError as e: # Catch JSON decode errors specifically
        print(f"      [Error] Failed to parse JSON response for '{title}'. API might be down or returning HTML.")
        print(f"      Response content preview: {response.text[:100]}")
    
    return ""

def segment_and_number(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'==.*?==', '', text) # Remove wiki headers
    sentences = re.split(r'\.\s+|\.$', text.strip())
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences if len(s.strip()) > 15]
    numbered = [f"[{i}] {s}" for i, s in enumerate(sentences)]
    return "\n".join(numbered)

def main():
    print("🌍 Fetching Cross-Domain Data (Malayalam Wikipedia)...")
    
    # 5 diverse topics: Kerala, ISRO, Artificial Intelligence, Mohanlal, Onam
    topics = ["കേരളം", "ഇസ്രോ", "നിർമ്മിത_ബുദ്ധി", "മോഹൻലാൽ", "ഓണം"]
    ood_data = []
    
    for idx, title in enumerate(topics):
        print(f"   Scraping '{title}'...")
        raw_text = fetch_malayalam_wikipedia(title)
        
        # Only process if we actually got text back
        if not raw_text:
            print(f"      [Skipping] No text retrieved for '{title}'.")
            continue
            
        # Take the first 15 sentences to keep it manageable
        sentences = segment_and_number(raw_text).split('\n')[:15]
        numbered_text = "\n".join(sentences)
        
        # We don't have human summaries for Wikipedia, so we use Oracle as ground truth
        ood_data.append({
            'Article_ID': f"WIKI_{idx}",
            'Numbered_Source_Text': numbered_text,
            'Original_Abstractive_Summary': "nan", # Will be filled by Oracle
        })

    if not ood_data:
        print("\n❌ Failed to scrape any articles. Check your network or Wikipedia API status.")
        return

    df = pd.DataFrame(ood_data)
    output_path = "data/wikipedia_ood_batch.csv"
    
    # Ensure the data directory exists
    import os
    os.makedirs("data", exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created Cross-Domain Dataset: {output_path}")
    print("Next step: You can manually annotate this, or run your Oracle on it to test Generalization!")

if __name__ == "__main__":
    main()