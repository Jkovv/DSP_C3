import pandas as pd
import numpy as np
import re
import zipfile
import random
import spacy
import torch
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

ZIP_PATH = 'data.zip'
TAXONOMY_PATH = 'taxonomie_df.csv' 
OUTPUT_FILE = 'final_hybrid_sbert_trainset.csv'
BATCH_SIZE = 64 # for sbert
SAMPLE_PCT = 1.0 # vs the 30% sample run (80%acc) 

print("Loading S-BERT model...")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
if torch.cuda.is_available():
    sbert_model = sbert_model.to('cuda')

try:
    spacy.prefer_gpu()
    nlp = spacy.load("nl_core_news_lg", disable=["parser"]) 
except:
    print("error: could not load spacy model.")
    exit()

def extract_numbers(text):
    if not isinstance(text, str) or pd.isna(text): return []
    pattern = r"\b(nul)\b|\b([a-zA-Z]*(twin|der|veer|vijf|zes|zeven|acht|negen)tig|[a-zA-Z]*tien|twee|drie|vier|vijf|zes|zeven|acht|negen|elf|twaalf)( )?(honderd|duizend|miljoen|miljard|procent)?\b|\b(honderd|duizend|miljoen|miljard)\b|\b[-+]?[.|,]?[\d]+(?:,\d\d\d)*[\.|,]?\d*([.|,]\d+)*(?:[eE][-+]?\d+)?( )?(honderd|duizend|miljoen|miljard|procent|%)?|half (miljoen|miljard|procent)"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    return list(set([m.group().strip().lower().replace('%', ' procent').replace(',', '.') for m in matches]))

def run_batched_preprocessing():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # spacy
        print("1/4 Lemmatizing Taxonomy...")
        tax_df = pd.read_csv(TAXONOMY_PATH)
        taxonomy_lemmas = set()
        for term in tqdm(tax_df.iloc[:, 0].str.lower().dropna()):
            doc = nlp(term)
            taxonomy_lemmas.add(" ".join([t.lemma_ for t in doc]))

        # spacy + sbert
        parent_path = [f for f in z.namelist() if 'all_parents.csv' in f][0]
        with z.open(parent_path) as f:
            df_parents = pd.read_csv(f).drop_duplicates(subset=['id']).dropna(subset=['content'])
        
        print("2/4 Pre-encoding Parents (CBS Reports)...")
        # spaCy features
        p_texts = df_parents['content'].str.lower().tolist()
        df_parents['p_docs'] = list(nlp.pipe(p_texts, batch_size=BATCH_SIZE))
        df_parents['p_lemmas'] = [{t.lemma_ for t in doc if t.is_alpha and not t.is_stop} for doc in df_parents['p_docs']]
        df_parents['p_numbers'] = df_parents['content'].apply(extract_numbers)
        df_parents['p_ents'] = [{ent.text.lower() for ent in doc.ents if ent.label_ in ['GPE', 'ORG']} for doc in df_parents['p_docs']]
        
        # S-BERT embeddings 
        p_embeddings = sbert_model.encode(p_texts, convert_to_tensor=True, show_progress_bar=True)
        
        # mapping
        parent_dict = df_parents.set_index('id').to_dict('index')
        parent_ids = df_parents['id'].tolist()
        id_to_idx = {p_id: i for i, p_id in enumerate(parent_ids)}

        child_files = [f for f in z.namelist() if 'data/c_' in f and f.endswith('.csv')]
        child_files.sort(key=lambda x: int(re.search(r'c_(\d+)', x).group(1)))
        child_files = child_files[:int(len(child_files) * SAMPLE_PCT)]
        
        child_to_parent_map = defaultdict(list)
        for _, p_row in df_parents.iterrows():
            c_ids = re.findall(r'\b\d+\b', str(p_row.get('related_children', '')))
            for cid in c_ids: child_to_parent_map[int(cid)].append(p_row['id'])

        final_results = []
        print(f"3/4 Processing {len(child_files)} articles...")
        
        for i in tqdm(range(0, len(child_files), BATCH_SIZE)):
            batch_list = child_files[i:i+BATCH_SIZE]
            batch_data = []

            for c_file in batch_list:
                try:
                    with z.open(c_file) as f:
                        df_c = pd.read_csv(f)
                    c_id = int(re.search(r'c_(\d+)', c_file).group(1))
                    batch_data.append({'id': c_id, 'text': " ".join(df_c['content'].astype(str))})
                except: continue

            c_texts = [d['text'].lower() for d in batch_data]
            c_docs = list(nlp.pipe(c_texts, batch_size=BATCH_SIZE))
            c_embeddings = sbert_model.encode(c_texts, convert_to_tensor=True)

            for idx, (data, c_doc) in enumerate(zip(batch_data, c_docs)):
                c_lemmas = {t.lemma_ for t in c_doc if t.is_alpha and not t.is_stop}
                c_nums = set(extract_numbers(data['text']))
                c_ents = {ent.text.lower() for ent in c_doc.ents if ent.label_ in ['GPE', 'ORG']}
                c_tax = c_lemmas.intersection(taxonomy_lemmas)
                
                target_ids = child_to_parent_map.get(data['id'], [])
                neg_id = random.choice([idx for idx in parent_ids if idx not in target_ids])
                
                for p_id, is_match in [(target_ids[0] if target_ids else None, 1), (neg_id, 0)]:
                    if p_id is None or p_id not in parent_dict: continue
                    p = parent_dict[p_id]
                    p_idx = id_to_idx[p_id]

                    sbert_sim = util.cos_sim(c_embeddings[idx], p_embeddings[p_idx]).item()
                    
                    final_results.append({
                        'child_id': data['id'],
                        'match': is_match,
                        'sbert_sim': sbert_sim,                                
                        'spacy_sim': c_doc.similarity(p['p_docs']),            
                        'tax_matches': len(c_tax.intersection(p['p_lemmas'])), 
                        'ner_overlap': len(c_ents & p['p_ents']),              
                        'num_matches': len(c_nums & set(p['p_numbers']))     
                    })

    pd.DataFrame(final_results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n4/4 Done! Hybrid dataset with S-BERT saved to: {OUTPUT_FILE}")

if __name__ == "__main__":

    run_batched_preprocessing()

