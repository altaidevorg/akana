import zeyrek, os, inspect, re

pkg_dir = os.path.dirname(inspect.getfile(zeyrek))
res_dir = os.path.join(pkg_dir, 'resources', 'tr')

dict_files = [
    'master-dictionary.dict',
    'non-tdk.dict',
    'locations-tr.dict',
    'person-names.dict',
    'proper-from-corpus.dict',
    'proper.dict',
    'abbreviations.dict',
    'informal.dict',
]

entries = {} # key = (lemma, root) -> (lemma, root, pos, sec_pos, attrs)

for fname in dict_files:
    fpath = os.path.join(res_dir, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Format: 'word [P:Pos; A:Attr1, Attr2]'
            match = re.match(r'^([^\s\[\]]+)(?:\s*\[(.*)\])?', line)
            if not match:
                continue
            raw_word = match.group(1).strip()
            meta = match.group(2) or ''
            
            if not raw_word or len(raw_word) == 0:
                continue
            
            # Ignore pure punctuation or symbols in root lexicon
            if not any(c.isalpha() for c in raw_word):
                continue

            lower_word = raw_word.replace('I', 'ı').replace('İ', 'i').lower()
            
            pos = 'Noun'
            sec_pos = 'None'
            attrs = set()
            
            # Determine POS from meta
            if 'P:Verb' in meta or lower_word.endswith('mak') or lower_word.endswith('mek'):
                pos = 'Verb'
            elif 'P:Adj' in meta:
                pos = 'Adj'
            elif 'P:Adv' in meta:
                pos = 'Adv'
            elif 'P:Pron' in meta:
                pos = 'Pron'
            elif 'P:Conj' in meta:
                pos = 'Conj'
            elif 'P:Postp' in meta:
                pos = 'Postp'
            elif 'P:Interj' in meta:
                pos = 'Interj'
            elif 'P:Ques' in meta:
                pos = 'Q'
            if 'P:ProperNoun' in meta or 'person-names' in fname or 'locations' in fname or 'proper' in fname:
                pos = 'Noun'
                sec_pos = 'ProperNoun'
                attrs.add('ProperNoun')
            
            if 'Voicing' in meta and 'NoVoicing' not in meta:
                attrs.add('Voicing')
            elif pos != 'Verb' and (lower_word.endswith('k') or lower_word.endswith('p') or lower_word.endswith('ç') or lower_word.endswith('t')):
                if 'NoVoicing' not in meta:
                    attrs.add('Voicing')
                    
            if 'VoicingNew' in meta or 'VoicingOpt' in meta:
                attrs.add('Voicing')
                
            if 'LastVowelDrop' in meta or 'VowelDrop' in meta:
                attrs.add('VowelDrop')
                
            if 'Doubling' in meta:
                attrs.add('Doubling')
                
            if 'InverseHarmony' in meta:
                attrs.add('InverseHarmony')
                
            root = lower_word
            if pos == 'Verb' and (lower_word.endswith('mak') or lower_word.endswith('mek')):
                root = lower_word[:-3]
                if root.endswith('t') or root.endswith('k') or root.endswith('p') or root.endswith('ç'):
                    attrs.add('Voicing')
                
            if 'proper-from-corpus' in fname:
                # If unvoiced version exists in master dictionary (e.g. kitab -> kitap, ağac -> ağaç), skip
                unvoiced = lower_word[:-1] + ({'b': 'p', 'c': 'ç', 'd': 't', 'ğ': 'k', 'g': 'k'}.get(lower_word[-1], lower_word[-1])) if len(lower_word) > 1 else lower_word
                if unvoiced != lower_word and ((unvoiced, unvoiced, 'Noun') in entries or (unvoiced, unvoiced, 'Adj') in entries):
                    continue

            key = (lower_word, root, pos)
            attr_str = ' '.join(sorted(list(attrs)))
            entries[key] = (lower_word, root, pos, sec_pos, attr_str)

# Also load 4600 words
with open('data/kalyoncu_words_4600.txt', encoding='utf-8') as f:
    for line in f:
        w = line.strip()
        if not w:
            continue
        if w.endswith('mak') or w.endswith('mek'):
            stem = w[:-3]
            attr = 'Voicing' if (stem.endswith('t') or stem.endswith('k') or stem.endswith('p') or stem.endswith('ç')) else ''
            entries[(w, stem, 'Verb')] = (w, stem, 'Verb', 'None', attr)
            entries[(stem, stem, 'Verb')] = (stem, stem, 'Verb', 'None', attr)
        else:
            attr = 'Voicing' if (w.endswith('k') or w.endswith('p') or w.endswith('ç') or w.endswith('t')) else ''
            entries[(w, w, 'Noun')] = (w, w, 'Noun', 'None', attr)

out_file = 'crates/akana-core/src/morphology/zemberek_lexicon.txt'
with open(out_file, 'w', encoding='utf-8') as f:
    for (lower_word, root, pos, sec_pos, attr_str) in sorted(entries.values()):
        if attr_str:
            f.write(f"{lower_word} {root} {pos} {sec_pos} {attr_str}\n")
        else:
            f.write(f"{lower_word} {root} {pos} {sec_pos}\n")

print(f"Generated {out_file} with {len(entries)} total entries.")
print(f"File size: {os.path.getsize(out_file) / 1024 / 1024:.2f} MB")
