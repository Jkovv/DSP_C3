import pandas as pd
import numpy as np
import re
import zipfile
import random
import spacy
from tqdm import tqdm
from collections import defaultdict

ZIP_PATH = 'data.zip'
TAXONOMY_PATH = 'taxonomie_df.csv' 
OUTPUT_FILE = 'actual_final_trainset_100pct_enhanced.csv' 
BATCH_SIZE = 50 

try:
    spacy.prefer_gpu()
    nlp = spacy.load("nl_core_news_lg", disable=["parser"]) 
except:
    print("error: could not load spacy model. Run: python -m spacy download nl_core_news_lg")
    exit()

def extract_numbers(text):
    if not isinstance(text, str) or pd.isna(text): return []
    pattern = r"\b(nul)\b|\b([a-zA-Z]*(twin|der|veer|vijf|zes|zeven|acht|negen)tig|[a-zA-Z]*tien|twee|drie|vier|vijf|zes|zeven|acht|negen|elf|twaalf)( )?(honderd|duizend|miljoen|miljard|procent)?\b|\b(honderd|duizend|miljoen|miljard)\b|\b[-+]?[.|,]?[\d]+(?:,\d\d\d)*[\.|,]?\d*([.|,]\d+)*(?:[eE][-+]?\d+)?( )?(honderd|duizend|miljoen|miljard|procent|%)?|half (miljoen|miljard|procent)"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    return list(set([m.group().strip().lower().replace('%', ' procent').replace(',', '.') for m in matches]))

def run_batched_preprocessing():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        print("Lemmatizing Taxonomy...")
        tax_df = pd.read_csv(TAXONOMY_PATH)
        raw_tax = tax_df.iloc[:, 0].str.lower().dropna().tolist()
        taxonomy_lemmas = set()
        for term in tqdm(raw_tax):
            doc = nlp(term)
            taxonomy_lemmas.add(" ".join([t.lemma_ for t in doc]))

        parent_path = [f for f in z.namelist() if 'all_parents.csv' in f][0]
        with z.open(parent_path) as f:
            df_parents = pd.read_csv(f).drop_duplicates(subset=['id']).dropna(subset=['content'])
        
        print("Processing parents...")
        # inverted triangle -> first 10 words should be ok as a title proxy 
        df_parents['title'] = df_parents['content'].apply(lambda x: " ".join(str(x).split()[:10]))
        
        p_docs = list(nlp.pipe(df_parents['content'].str.lower(), batch_size=BATCH_SIZE))
        p_titles = list(nlp.pipe(df_parents['title'].str.lower(), batch_size=BATCH_SIZE))
        
        df_parents['clean_doc'] = p_docs
        df_parents['title_doc'] = p_titles
        df_parents['p_lemmas'] = [{t.lemma_ for t in doc if t.is_alpha and not t.is_stop} for doc in p_docs]
        df_parents['p_numbers'] = df_parents['content'].apply(extract_numbers)
        df_parents['p_ents'] = [{ent.text.lower() for ent in doc.ents if ent.label_ in ['GPE', 'ORG']} for doc in p_docs]
        df_parents['p_tax_matches'] = df_parents['p_lemmas'].apply(lambda x: x.intersection(taxonomy_lemmas))
        df_parents['content_len'] = df_parents['content'].apply(lambda x: len(str(x).split()))

        # mapping
        child_to_parent_map = defaultdict(list)
        for _, p_row in df_parents.iterrows():
            c_ids = re.findall(r'\b\d+\b', str(p_row.get('related_children', '')))
            for cid in c_ids: child_to_parent_map[int(cid)].append(p_row['id'])
        
        parent_dict = df_parents.set_index('id').to_dict('index')
        parent_ids = df_parents['id'].tolist()

        child_files = [f for f in z.namelist() if 'data/c_' in f and 'output' not in f and f.endswith('.csv')]
        child_files.sort(key=lambda x: int(re.search(r'c_(\d+)', x).group(1)))
        
        child_files = child_files[:int(len(child_files) * 1)] # we can change this number for subsample
        
        print(f"Processing {len(child_files)} child articles...")
        final_results = []
        
        for i in tqdm(range(0, len(child_files), BATCH_SIZE), desc="Progress"):
            batch_list = child_files[i:i+BATCH_SIZE]
            batch_data = []

            for c_file in batch_list:
                try:
                    with z.open(c_file) as f:
                        df_c = pd.read_csv(f)
                    c_id = int(re.search(r'c_(\d+)', c_file).group(1))
                    content = " ".join(df_c['content'].astype(str).tolist())
                    title = str(df_c.iloc[0].get('title', content[:100]))
                    batch_data.append({'id': c_id, 'text': content, 'title': title})
                except: continue

            c_docs = list(nlp.pipe([d['text'].lower() for d in batch_data], batch_size=BATCH_SIZE))
            c_titles = list(nlp.pipe([d['title'].lower() for d in batch_data], batch_size=BATCH_SIZE))

            for data, c_doc, c_title_doc in zip(batch_data, c_docs, c_titles):
                c_lemmas = {t.lemma_ for t in c_doc if t.is_alpha and not t.is_stop}
                c_nums = set(extract_numbers(data['text']))
                c_ents = {ent.text.lower() for ent in c_doc.ents if ent.label_ in ['GPE', 'ORG']}
                c_tax_matches = c_lemmas.intersection(taxonomy_lemmas)
                c_len = len(data['text'].split())
                
                target_ids = child_to_parent_map.get(data['id'], [])
                for p_id in target_ids:
                    if p_id not in parent_dict: continue
                    p = parent_dict[p_id]
                    
                    # pos pair
                    final_results.append({
                        'child_id': data['id'], 'match': 1,
                        'vector_similarity': c_doc.similarity(p['clean_doc']),
                        'title_similarity': c_title_doc.similarity(p['title_doc']),
                        'numbers_lenmatches': len(c_nums & set(p['p_numbers'])),
                        'taxonomy_lenmatches': len(c_tax_matches.intersection(p['p_tax_matches'])),
                        'ner_overlap': len(c_ents & p['p_ents']),
                        'length_ratio': min(c_len, p['content_len']) / max(c_len, p['content_len'])
                    })

                    # neg pair
                    neg_id = random.choice([idx for idx in parent_ids if idx not in target_ids])
                    neg_p = parent_dict[neg_id]
                    final_results.append({
                        'child_id': data['id'], 'match': 0,
                        'vector_similarity': c_doc.similarity(neg_p['clean_doc']),
                        'title_similarity': c_title_doc.similarity(neg_p['title_doc']),
                        'numbers_lenmatches': len(c_nums & set(neg_p['p_numbers'])),
                        'taxonomy_lenmatches': len(c_tax_matches.intersection(neg_p['p_tax_matches'])),
                        'ner_overlap': len(c_ents & neg_p['p_ents']),
                        'length_ratio': min(c_len, neg_p['content_len']) / max(c_len, neg_p['content_len'])
                    })

    pd.DataFrame(final_results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nPreprocessing complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_batched_preprocessing()