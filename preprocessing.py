import pandas as pd
import numpy as np
import re
import zipfile
import random
import torch
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

ZIP_PATH = 'data.zip'
OUTPUT_FILE = 'final_hybrid_sbert_trainset_100pct.csv' 
NEG_SAMPLES_COUNT = 24
BATCH_SIZE = 64

print("Loading Multilingual S-BERT model...")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
if torch.cuda.is_available(): sbert_model = sbert_model.to('cuda')

def run_hybrid_preprocessing():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        parent_file = next(f for f in z.namelist() if f.endswith('all_parents.csv'))
        with z.open(parent_file) as f:
            df_parents = pd.read_csv(f).drop_duplicates(subset=['id']).dropna(subset=['content'])
        
        print("Encoding Parent Reports (S-BERT)...")
        p_embeddings = sbert_model.encode(df_parents['content'].str.lower().tolist(), 
                                          convert_to_tensor=True, show_progress_bar=True)
        parent_ids = df_parents['id'].tolist()
        
        child_to_parent_map = defaultdict(list)
        for _, row in df_parents.iterrows():
            c_ids = re.findall(r'\d+', str(row.get('related_children', '')))
            for cid in c_ids:
                child_to_parent_map[int(cid)].append(row['id'])

        child_files = [f for f in z.namelist() if 'data/c_' in f and f.endswith('.csv')]
        final_results = []

        print(f"Processing {len(child_files)} articles...")
        for i in tqdm(range(0, len(child_files), BATCH_SIZE)):
            batch = child_files[i:i+BATCH_SIZE]
            batch_data = []
            for c_file in batch:
                try:
                    with z.open(c_file) as f:
                        c_id = int(re.search(r'c_(\d+)', c_file).group(1))
                        batch_data.append({'id': c_id, 'text': " ".join(pd.read_csv(f)['content'].astype(str))})
                except: continue

            c_embeddings = sbert_model.encode([d['text'].lower() for d in batch_data], convert_to_tensor=True)

            for idx, data in enumerate(batch_data):
                targets = child_to_parent_map.get(data['id'], [])
                if not targets: continue
                
                potential_negs = [p for p in parent_ids if p not in targets]
                neg_ids = random.sample(potential_negs, min(len(potential_negs), NEG_SAMPLES_COUNT))

                # ranking pairs
                for p_id, is_match in [(targets[0], 1)] + [(n, 0) for n in neg_ids]:
                    p_idx = parent_ids.index(p_id)
                    similarity = util.cos_sim(c_embeddings[idx], p_embeddings[p_idx]).item()
                    
                    final_results.append({
                        'child_id': data['id'],
                        'parent_id': p_id,
                        'match': is_match,
                        'sbert_sim': similarity
                    })

    pd.DataFrame(final_results).to_csv(OUTPUT_FILE, index=False)
    print(f"Hybrid dataset saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_hybrid_preprocessing()
