import pandas as pd
import numpy as np
import re
import zipfile
import random
import torch
import nltk
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
DUTCH_STOPWORDS = set(stopwords.words('dutch'))

ZIP_PATH = '/content/data.zip'
TAXONOMY_PATH = 'taxonomie_df.csv'
OUTPUT_FILE = 'final_hybrid_sbert_trainset_100pct.csv'
NEG_SAMPLES_COUNT = 24 

print("Loading S-BERT model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
if torch.cuda.is_available(): model = model.to('cuda')

def nltk_clean_text(text):
    """Standardizes and tokenizes Dutch text using NLTK."""
    if not isinstance(text, str) or pd.isna(text): return set()
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = word_tokenize(text)
    clean_tokens = [t for t in tokens if t not in DUTCH_STOPWORDS and len(t) > 2]
    return set(clean_tokens)



def extract_numbers_complex(text):
    if not isinstance(text, str) or pd.isna(text): return set()
    pattern = r"\b(nul)\b|\b([a-zA-Z]*(twin|der|veer|vijf|zes|zeven|acht|negen)tig|[a-zA-Z]*tien|twee|drie|vier|vijf|zes|zeven|acht|negen|elf|twaalf)( )?(honderd|duizend|miljoen|miljard|procent)?\b|\b(honderd|duizend|miljoen|miljard)\b|\b[-+]?[.|,]?[\d]+(?:,\d\d\d)*[\.|,]?\d*([.|,]\d+)*(?:[eE][-+]?\d+)?( )?(honderd|duizend|miljoen|miljard|procent|%)?|half (miljoen|miljard|procent)"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    return set([m.group().strip().lower().replace('%', ' procent').replace(',', '.') for m in matches])

def run_hybrid_preprocessing():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        print("1/4 Processing Taxonomy...")
        tax_df = pd.read_csv(TAXONOMY_PATH)
        taxonomy_terms = set()
        for term in tax_df.iloc[:, 1].dropna().astype(str):
            taxonomy_terms.update(nltk_clean_text(term))

        print("2/4 Encoding Parents...")
        p_path = [f for f in z.namelist() if f.endswith('all_parents.csv')][0]
        with z.open(p_path) as f:
            df_p = pd.read_csv(f).drop_duplicates(subset=['id']).dropna(subset=['content']).reset_index(drop=True)
        
        p_embeddings = model.encode(df_p['content'].str.lower().tolist(), convert_to_tensor=True, show_progress_bar=True)
        
        p_info = {}
        child_to_p = defaultdict(list)
        for i, row in df_p.iterrows():
            content = str(row['content'])
            word_set = nltk_clean_text(content)
            p_info[row['id']] = {
                'emb_idx': i,
                'words': word_set,
                'nums': extract_numbers_complex(content),
                'tax_words': word_set.intersection(taxonomy_terms)
            }
            for cid in re.findall(r'\d+', str(row.get('related_children', ''))):
                child_to_p[int(cid)].append(row['id'])

        p_ids = list(p_info.keys())

        print("3/4 Processing Children and 1:24 negative sampling...")
        c_files = [f for f in z.namelist() if 'data/c_' in f and f.endswith('.csv')]
        final_data = []

        for cf in tqdm(c_files):
            try:
                c_id = int(re.search(r'c_(\d+)', cf).group(1))
                if c_id not in child_to_p: continue
                
                with z.open(cf) as f:
                    c_text_raw = " ".join(pd.read_csv(f)['content'].astype(str))
                
                c_words = nltk_clean_text(c_text_raw)
                c_nums = extract_numbers_complex(c_text_raw)
                c_tax = c_words.intersection(taxonomy_terms)
                c_emb = model.encode(c_text_raw.lower(), convert_to_tensor=True)

                target_id = child_to_p[c_id][0]
                if target_id not in p_info: continue

                negs = random.sample([pid for pid in p_ids if pid != target_id], NEG_SAMPLES_COUNT)
                
                for p_id, match in [(target_id, 1)] + [(n, 0) for n in negs]:
                    p = p_info[p_id]
                    sim = util.cos_sim(c_emb, p_embeddings[p['emb_idx']]).item()
                    common_words = c_words.intersection(p['words'])
                    jaccard = len(common_words) / len(c_words.union(p['words'])) if len(c_words.union(p['words'])) > 0 else 0

                    final_data.append({
                        'child_id': c_id,
                        'parent_id': p_id,
                        'match': match,
                        'sbert_sim': sim,
                        'word_jaccard_sim': jaccard,
                        'tax_overlap_count': len(c_tax.intersection(p['tax_words'])),
                        'num_overlap_count': len(c_nums.intersection(p['nums'])),
                        'common_word_count': len(common_words)
                    })
            except Exception: continue

        pd.DataFrame(final_data).to_csv(OUTPUT_FILE, index=False)
        print(f"\n4/4 Success! Hybrid file (1:24) saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_hybrid_preprocessing()
