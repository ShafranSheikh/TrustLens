#!/usr/bin/env python3
"""
Build the TrustLens gold dataset.

SOURCE OF TRUTH : Dataset/LabelledDataset/Labellled and verified/Labelled_Verified Dataset.xlsx
                  (manual annotation - the only source of label VALUES)
STRUCTURE FROM  : annotated_reviews_v2.csv (review_id scheme and column layout only)

PRINCIPLE: this script never overwrites a label that was annotated. It only
removes out-of-scope rows, fixes spelling/case, fills blanks where a structural
rule applies, and adds derived columns. Every change is reported on stdout.

Optional: Dataset/Gold/manual_label_fixes.csv - review_id plus any of
language_type / sentiment_polarity / authenticity. Values there fill BLANKS only.

Run:  python3 scripts/build_gold_dataset.py
"""
import re, json, unicodedata, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT/'Dataset'/'LabelledDataset'/'Labellled and verified'/'Labelled_Verified Dataset.xlsx'
OUT  = ROOT/'Dataset'/'Gold'; OUT.mkdir(parents=True, exist_ok=True)
FIXES = OUT/'manual_label_fixes.csv'

def note(m, n=None): print(f'  {m}' + (f'  [{n}]' if n is not None else ''))

# rows removed by hand-verified decision, with the reason recorded
DROP = {'R1142':'romanised Tamil - outside English/Sinhala/Singlish scope',
        'R1311':'romanised Tamil - outside English/Sinhala/Singlish scope',
        'R1845':'not a review - a date value leaked into the review_text cell'}
# non-linguistic strings the automatic rule cannot catch (they contain letters)
EXTRA_NON_VERBAL = ['R0509','R1213','R1399','R1590','R1829']

# ---------------------------------------------------------------- load
df = pd.read_excel(SRC, dtype=str).fillna('')
note(f'loaded {SRC.name}', len(df))
df['review_id'] = [f'R{i:04d}' for i in range(2, len(df)+2)]
rows_in = len(df)

def clean_date(s):
    m = re.match(r'(\d{4}-\d{2}-\d{2})', str(s).strip()); return m.group(1) if m else str(s).strip()
def clean_int(s):
    try: return str(int(float(str(s).strip())))
    except: return str(s).strip()
df['date']=df['date'].map(clean_date)
df['star_rating']=df['star_rating'].map(clean_int)
df['helpful_count']=df['helpful_count'].map(clean_int)
for c in ['language_type','sentiment_polarity','authenticity','category','aspects']:
    df[c]=df[c].astype(str).str.strip()
note('normalised date / star_rating / helpful_count formats')

# ---------------------------------------------------------------- row removals
TAMIL = lambda t: any('஀'<=ch<='௿' for ch in str(t))
removed = []
RULES = [('empty',  'empty review_text'),
         ('tamil',  'Tamil script - outside scope'),
         ('listed', 'see reason column')]
for kind, reason in RULES:
    # recompute against the CURRENT frame each pass, never a stale mask
    if   kind=='empty':  mask = df['review_text'].astype(str).str.strip()==''
    elif kind=='tamil':  mask = df['review_text'].map(TAMIL)
    else:                mask = df['review_id'].isin(DROP)
    hit = df[mask]
    for _,r in hit.iterrows():
        removed.append({'review_id':r['review_id'],
                        'reason': DROP.get(r['review_id'], reason),
                        'review_text': str(r['review_text'])[:200]})
    df = df[~mask]
    note(f'removed: {reason}', len(hit))
note('rows remaining', len(df))

# ---------------------------------------------------------------- vocabulary fixes
n_fake = int((df['authenticity']=='Fake').sum())
df['authenticity'] = df['authenticity'].replace({'Fake':'Suspicious'})
note("authenticity: 'Fake' -> 'Suspicious'", n_fake)

SPELLING = {'color':'Color/Design','size':'Size/Fit'}
MERGE = {'Product Quality':'Product Quality','Quality':'Product Quality',
         'Material/Quality':'Product Quality','Battery/Performance':'Product Quality',
         'Expiry/Safety':'Product Quality','Item Authenticity':'Product Quality',
         'Delivery/Shipping':'Delivery/Shipping','Price/Value':'Price/Value',
         'Seller Service':'Seller Service','Color/Design':'Color/Design',
         'Packaging':'Packaging','Size/Fit':'Size/Fit','Other':None}
def parse(s):
    out=[]
    for p in str(s).split(';'):
        p=p.strip()
        if not p: continue
        if ':' in p: c,pol=p.rsplit(':',1); out.append((c.strip(),pol.strip()))
        else: out.append((p,''))
    return out
def render(ps): return '; '.join(f'{c}:{p}' if p else c for c,p in ps)
fixed=0; dropped_other=0; ac=[]; am=[]
for s in df['aspects']:
    a=[]
    for c,p in parse(s):
        if c in SPELLING: c=SPELLING[c]; fixed+=1
        a.append((c,p))
    ac.append(render(a))
    m=[]
    for c,p in a:
        t=MERGE.get(c,c)
        if t is None: dropped_other+=1; continue
        if (t,p) not in m: m.append((t,p))
    am.append(render(m))
df['aspects']=ac; df['aspects_merged']=am
note("aspect spelling fixed ('color'/'size')", fixed)
note("aspects_merged: 7 categories; 'Other' dropped", dropped_other)

# ---------------------------------------------------------------- non-verbal reviews
def strip_sym(s):
    return ''.join(ch for ch in str(s) if not (
        unicodedata.category(ch) in ('So','Sk','Cf') or '\U0001F000'<=ch<='\U0001FAFF'
        or '☀'<=ch<='➿' or ch in '️‍⭐❤'))
# a review with no Latin and no Sinhala letter once symbols are removed carries no language
rule = ~df['review_text'].map(lambda t: bool(re.search(r'[A-Za-z඀-෿]', strip_sym(t))))
nonverbal = rule | df['review_id'].isin(EXTRA_NON_VERBAL)
df['text_type'] = nonverbal.map({True:'non-verbal', False:'linguistic'})
note('flagged as non-verbal (emoji-only or keyboard mash)', int(nonverbal.sum()))

# fill language_type ONLY where it is blank - never overwrite an annotated value
fill = nonverbal & (df['language_type']=='')
df.loc[fill,'language_type']='Non-verbal'
note("language_type set to 'Non-verbal' where it was blank", int(fill.sum()))
kept = df[nonverbal & (df['language_type']!='Non-verbal')]
if len(kept):
    note(f'non-verbal rows keeping their annotated language: {sorted(kept["review_id"])}')

# ---------------------------------------------------------------- optional manual fixes
ALLOWED = {
    'language_type':      {'English','Sinhala','Singlish','Mixed','Non-verbal'},
    'sentiment_polarity': {'Positive','Neutral','Negative','Not applicable'},
    'authenticity':       {'Genuine','Suspicious'},
}
applied=0
if FIXES.exists():
    fx = pd.read_csv(FIXES, dtype=str).fillna('')
    fx = fx[fx['review_id'].astype(str).str.strip()!='']
    idx = df.set_index('review_id')
    for _,r in fx.iterrows():
        rid=str(r['review_id']).strip()
        if rid not in idx.index: continue
        for col in ['language_type','sentiment_polarity','authenticity']:
            v=str(r.get(col,'')).strip()
            if not v: continue
            if v not in ALLOWED[col]:
                raise ValueError(f'{FIXES.name}: {rid} has {col}={v!r}; '
                                 f'allowed values are {sorted(ALLOWED[col])}')
            if idx.at[rid,col]=='':                # blanks only, never an overwrite
                df.loc[df['review_id']==rid, col]=v; applied+=1
    note(f'applied manual label fixes from {FIXES.name}', applied)
else:
    note(f'{FIXES.name} not present - skipping manual fixes')

# ---------------------------------------------------------------- derived columns
df['n_chars']=df['review_text'].astype(str).str.len()
norm=df['review_text'].astype(str).str.replace(r'\s+',' ',regex=True).str.strip().str.lower()
df['dup_count']=norm.map(norm.value_counts())
# Group on normalised TEXT ALONE. Grouping on product+text is infeasible here:
# generic strings ("good") appear on nearly every product, so the product+text
# graph collapses into one component holding 95.5% of the corpus and no test set
# can be carved out. Text leakage is the serious one for a text classifier;
# product overlap is recorded as a limitation and checked separately by the
# unseen-product holdout in trustlens.splits.
df['group_key']=norm.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:12])
df['has_labels']=((df['language_type']!='')&(df['sentiment_polarity']!='')&
                  (df['authenticity']!='')).map({True:'yes',False:'no'})

COLS=['review_id','category','product_name','date','verified_purchase','star_rating',
      'review_text','variant','helpful_count','language_type','sentiment_polarity',
      'aspects','aspects_merged','authenticity','text_type','n_chars','dup_count',
      'group_key','has_labels']
gold=df[COLS].copy()
gold.to_csv(OUT/'annotated_reviews_gold_v1.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(removed).to_csv(OUT/'removed_rows.csv', index=False, encoding='utf-8-sig')

# template for the outstanding manual labels
if not FIXES.exists():
    todo=gold[gold['has_labels']=='no'][['review_id','review_text','star_rating','language_type',
                                         'sentiment_polarity','authenticity','text_type']]
    todo.to_csv(OUT/'manual_label_fixes_TEMPLATE.csv', index=False, encoding='utf-8-sig')

print('\n=== GOLD FILE ===  rows:',len(gold))
for c in ['language_type','sentiment_polarity','authenticity','text_type']:
    print(f'  {c}: {gold[c].value_counts(dropna=False).to_dict()}')
print('  fully labelled:',(gold['has_labels']=='yes').sum(),'/',len(gold),
      ' | still incomplete:',(gold['has_labels']=='no').sum())
print('  linguistic rows:',(gold['text_type']=='linguistic').sum(),
      '| split groups:',gold['group_key'].nunique())
(OUT/'build_summary.json').write_text(json.dumps(
    {'source':SRC.name,'rows_in':rows_in,'rows_out':len(gold),'removed':len(removed),
     'fake_to_suspicious':n_fake,'aspect_spelling_fixed':fixed,
     'non_verbal':int(nonverbal.sum()),'manual_fixes_applied':applied}, indent=2))
print('\nwritten: annotated_reviews_gold_v1.csv | removed_rows.csv | build_summary.json'
      + ('' if FIXES.exists() else ' | manual_label_fixes_TEMPLATE.csv'))
