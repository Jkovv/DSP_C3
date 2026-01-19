import pandas as pd
import numpy as np
import re
import zipfile
import random
from tqdm import tqdm
from collections import defaultdict

ZIP_PATH = 'data.zip'
TAXONOMY_PATH = 'taxonomie_df.csv' 
OUTPUT_FILE = 'final_basic_trainset_fixed.csv' 

def clean_text_to_set(text):
    if not isinstance(text, str) or pd.isna(text): return set()
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return set(text.split())

def extract_numbers(text):
    if not isinstance(text, str) or pd.isna(text): return []
    pattern = r"\b(nul)\b|\b([a-zA-Z]*(twin|der|veer|vijf|zes|zeven|acht|negen)tig|[a-zA-Z]*tien|twee|drie|vier|vijf|zes|zeven|acht|negen|elf|twaalf)( )?(honderd|duizend|miljoen|miljard|procent)?\b|\b(honderd|duizend|miljoen|miljard)\b|\b[-+]?[.|,]?[\d]+(?:,\d\d\d)*[\.|,]?\d*([.|,]\d+)*(?:[eE][-+]?\d+)?( )?(honderd|duizend|miljoen|miljard|procent|%)?|half (miljoen|miljard|procent)"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    return list(set([m.group().strip().lower().replace('%', ' procent').replace(',', '.') for m in matches]))

def run_basic_preprocessing():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        print("1/4 Loading Taxonomy...")
        tax_df = pd.read_csv(TAXONOMY_PATH)
        taxonomy_words = set()
        for term in tax_df.iloc[:, 0].str.lower().dropna():
            taxonomy_words.update(clean_text_to_set(term))

        parent_path = [f for f in z.namelist() if 'all_parents.csv' in f][0]
        with z.open(parent_path) as f:
            df_parents = pd.read_csv(f).drop_duplicates(subset=['id']).dropna(subset=['content'])
        
        parent_dict = {}
        parent_ids = df_parents['id'].tolist()
        for _, row in tqdm(df_parents.iterrows(), total=len(df_parents)):
            content = str(row['content'])
            word_set = clean_text_to_set(content)
            parent_dict[row['id']] = {
                'words': word_set,
                'numbers': set(extract_numbers(content)),
                'tax_words': word_set.intersection(taxonomy_words)
            }

        child_to_parent_map = defaultdict(list)
        for _, p_row in df_parents.iterrows():
            c_ids = re.findall(r'\b\d+\b', str(p_row.get('related_children', '')))
            for cid in c_ids: child_to_parent_map[int(cid)].append(p_row['id'])

        child_files = [f for f in z.namelist() if 'data/c_' in f and f.endswith('.csv')]
        child_files.sort(key=lambda x: int(re.search(r'c_(\d+)', x).group(1)))

        final_results = []
        print(f"3/4 Processing {len(child_files)} articles...")
        for c_file in tqdm(child_files):
            try:
                with z.open(c_file) as f:
                    df_c = pd.read_csv(f)
                    c_id = int(re.search(r'c_(\d+)', c_file).group(1))
                    text = " ".join(df_c['content'].astype(str))
            except: continue

            c_words = clean_text_to_set(text)
            c_nums = set(extract_numbers(text))
            c_tax_words = c_words.intersection(taxonomy_words)
            target_ids = child_to_parent_map.get(c_id, [])
            neg_id = random.choice([pid for pid in parent_ids if pid not in target_ids])
            
            for p_id, is_match in [(target_ids[0] if target_ids else None, 1), (neg_id, 0)]:
                if p_id is None or p_id not in parent_dict: continue
                p = parent_dict[p_id]
                intersection = len(c_words.intersection(p['words']))
                union = len(c_words.union(p['words']))
                jaccard_sim = intersection / union if union > 0 else 0

                final_results.append({
                    'child_id': c_id,
                    'parent_id': p_id,  # ADDED
                    'match': is_match,
                    'word_jaccard_sim': jaccard_sim,
                    'tax_overlap_count': len(c_tax_words.intersection(p['tax_words'])),
                    'num_overlap_count': len(c_nums.intersection(p['numbers'])),
                    'common_word_count': intersection
                })

    pd.DataFrame(final_results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n4/4 Done! Basic dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_basic_preprocessing()
