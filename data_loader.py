"""
data _loader.py — Dataset loading and dependency parsing for English & Russian.

Provides subject-verb agreement data with dependency metadata
parsed via stanza for the Syntactic Resilience experiment.

COLAB FIXES:
  - Added max_sentences default of 500 to avoid Colab RAM/time limits
  - Added max_length=64 cap on tokenizer to avoid OOM
  - Wrapped stanza download in robust try/except
  - BLiMP dataset loaded with trust_remote_code=True for newer HF versions
  - RuCoLA fallback: tries HuggingFace hub if local CSV not found
  - nsubj:pass also captured (passive constructions)
  - Tokenizer given explicit max_length and padding side
  - Fixed: stanza import moved to top (not inside function)
"""

import os
import json
import random

import numpy as np
import torch

import stanza
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from datasets import load_dataset


# ── Token-alignment helper ────────────────────────────────────────────────────

def map_word_to_token(char_start, offset_mapping):
    """
    Map a word's character start position to its first subword token index.
    Stanza returns word-level indices; GPT-2 uses subword tokens — this bridges
    the gap so syntax-match checks look at the correct token positions.
    """
    if char_start is None or not offset_mapping:
        return None
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start <= char_start < tok_end:
            return tok_idx
    # Fallback: nearest token start
    return min(range(len(offset_mapping)),
               key=lambda i: abs(offset_mapping[i][0] - char_start))


# ── Dependency parsing ────────────────────────────────────────────────────────

def _parse_dependencies(sentences, nlp):
    """
    Parse sentences with stanza and extract subject-verb dependency pairs.
    Now also captures character offsets so token alignment can be done later.
    """
    parsed = []
    for sent_text in sentences:
        dep_info = {
            'text':               sent_text,
            'subject_idx':        None,   # Stanza word index (0-based)
            'verb_idx':           None,   # Stanza word index (0-based)
            'subject_char_start': None,   # character offset of subject word
            'verb_char_start':    None,   # character offset of verb word
            'is_plural':          None,
            'sentence_length':    len(sent_text.split()),
        }
        try:
            doc = nlp(sent_text)
            for sent in doc.sentences:
                word_map = {w.id: w for w in sent.words}
                for word in sent.words:
                    if word.deprel in ('nsubj', 'nsubj:pass'):
                        dep_info['subject_idx']        = word.id - 1
                        dep_info['subject_char_start'] = word.start_char
                        feats = word.feats if word.feats else ''
                        dep_info['is_plural'] = 1 if 'Number=Plur' in feats else 0
                        dep_info['verb_idx']           = word.head - 1
                        head = word_map.get(word.head)
                        if head is not None:
                            dep_info['verb_char_start'] = head.start_char
                        break
                if dep_info['subject_idx'] is not None:
                    break
        except Exception:
            pass
        parsed.append(dep_info)
    return parsed


# ── Main loader ───────────────────────────────────────────────────────────────

def load_agreement_dataset(lang='en',
                           max_sentences=500,
                           tokenizer_name='gpt2-medium',
                           max_length=64):
    """
    Load subject-verb agreement dataset with dependency metadata.

    Args:
        lang           : 'en' for English, 'ru' for Russian.
        max_sentences  : cap on sentences (default 500 for Colab safety).
        tokenizer_name : HuggingFace tokenizer name.
        max_length     : tokeniser max_length (capped at 64 for Colab RAM).

    Returns:
        inputs         : dict with 'input_ids' and 'attention_mask' tensors.
        labels         : numpy array of binary labels (0=singular, 1=plural).
        dep_metadata   : list of dicts with dependency info per sentence.
        sentence_types : list of dicts with 'length'/'complexity' tags.
    """
    # ── Load raw sentences ────────────────────────────────────────────
    if lang == 'en':
        print("  Loading BLiMP dataset from HuggingFace...")
        sentences = []
        for subset in [
            "regular_plural_subject_verb_agreement_1",
            "regular_plural_subject_verb_agreement_2",
        ]:
            try:
                ds = load_dataset("blimp", subset, split="train")
            except Exception:
                ds = load_dataset("blimp", subset, split="train",
                                  trust_remote_code=False)
            sentences += [item['sentence_good'] for item in ds]

        random.seed(42)
        random.shuffle(sentences)

    elif lang == 'ru':
        print("  Loading RuCoLA dataset...")
        sentences = []

        # First try: local CSV (rucola_train.csv in working directory)
        local_csv = 'rucola_train.csv'
        if os.path.exists(local_csv):
            print(f"    Found local file: {local_csv}")
            try:
                ds = load_dataset('csv', data_files=local_csv, split='train')
                sentences = [
                    item['sentence'] for item in ds
                    if item.get('acceptable') == 1
                ]
            except Exception as e:
                print(f"    Local CSV load failed: {e}")

        # Fallback: try HuggingFace hub
        if not sentences:
            print("    Trying HuggingFace hub for RuCoLA...")
            try:
                ds = load_dataset('RussianNLP/rucola', split='train',
                                  trust_remote_code=True)
                sentences = [
                    item['sentence'] for item in ds
                    if item.get('acceptable') == 1
                ]
            except Exception as e:
                print(f"    HF hub load failed: {e}")

        if not sentences:
            raise RuntimeError(
                "Could not load Russian data. "
                "Place rucola_train.csv in your Colab working directory "
                "or ensure HuggingFace access to RussianNLP/rucola."
            )

        random.seed(42)
        random.shuffle(sentences)

    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'en' or 'ru'.")

    # ── Cap sentence count ────────────────────────────────────────────
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    else:
        print("  WARNING: max_sentences=None. Stanza parsing will take very "
              "long on Colab. Consider setting max_sentences=500.")

    # ── Stanza parse with caching ─────────────────────────────────────
    cache_path = f'stanza_cache_{lang}.json'
    if os.path.exists(cache_path):
        print(f"  Loading Stanza cache from {cache_path}...")
        with open(cache_path) as f:
            cache_map = json.load(f)
        dep_metadata = []
        for s in sentences:
            if s in cache_map:
                dep_metadata.append(cache_map[s])
            else:
                dep_metadata.append({
                    'text': s, 'subject_idx': None, 'verb_idx': None,
                    'is_plural': None, 'sentence_length': len(s.split()),
                })
    else:
        print(f"  Initializing stanza pipeline for '{lang}'...")
        try:
            stanza.download(lang,
                            processors='tokenize,pos,lemma,depparse',
                            verbose=False)
        except Exception:
            pass  # Already downloaded, or network issue — pipeline init will tell us

        nlp = stanza.Pipeline(
            lang,
            processors='tokenize,pos,lemma,depparse',
            verbose=False,
            use_gpu=torch.cuda.is_available(),
        )

        print(f"  Parsing {len(sentences)} sentences...")
        dep_metadata = _parse_dependencies(sentences, nlp)

        cache_map = {meta['text']: meta for meta in dep_metadata}
        print(f"  Saving Stanza cache to {cache_path}...")
        with open(cache_path, 'w') as f:
            json.dump(cache_map, f)

    # ── Filter to sentences with valid nsubj ──────────────────────────
    labels, valid_sentences, valid_metadata = [], [], []
    for i, meta in enumerate(dep_metadata):
        if meta['is_plural'] is not None:
            labels.append(meta['is_plural'])
            valid_sentences.append(sentences[i])
            valid_metadata.append(meta)

    labels = np.array(labels)
    print(f"  {len(valid_sentences)} sentences with valid nsubj dependencies found.")

    if len(valid_sentences) == 0:
        raise RuntimeError(
            "No valid sentences found after dependency parsing. "
            "Check stanza models or your input data."
        )

    # ── Token alignment: word indices → GPT-2 subword token indices ───────────
    # Stanza gives word-level positions; GPT-2 splits words into subword tokens.
    # Without this step, syntax-match checks the wrong token positions → 0 scores.
    print("  Aligning Stanza word indices to GPT-2 subword token indices...")
    try:
        _fast_tok = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        _fast_tok.pad_token = _fast_tok.eos_token
        n_aligned = 0
        for meta in valid_metadata:
            s_char = meta.get('subject_char_start')
            v_char = meta.get('verb_char_start')
            if s_char is not None:
                enc = _fast_tok(meta['text'],
                                return_offsets_mapping=True,
                                add_special_tokens=False)
                om = enc['offset_mapping']
                meta['subject_token_idx'] = map_word_to_token(s_char, om)
                meta['verb_token_idx']    = map_word_to_token(v_char, om)
                n_aligned += 1
            else:
                # Cached entries from before this fix: fall back to word index
                meta['subject_token_idx'] = meta.get('subject_idx')
                meta['verb_token_idx']    = meta.get('verb_idx')
            # Final safety fallback
            if meta['subject_token_idx'] is None:
                meta['subject_token_idx'] = meta.get('subject_idx')
            if meta['verb_token_idx'] is None:
                meta['verb_token_idx'] = meta.get('verb_idx')
        print(f"  Token alignment done ({n_aligned}/{len(valid_metadata)} sentences had char offsets).")
    except Exception as e:
        print(f"  [WARNING] Token alignment failed ({e}). Using word indices as fallback.")
        for meta in valid_metadata:
            meta.setdefault('subject_token_idx', meta.get('subject_idx'))
            meta.setdefault('verb_token_idx',    meta.get('verb_idx'))

    # ── Sentence type tags ────────────────────────────────────────────
    sentence_types = []
    rel_words_en = ['who ', 'which ', 'that ']
    rel_words_ru = ['который', 'которая', 'которые', 'которое']
    for meta in valid_metadata:
        length_tag = 'short' if meta['sentence_length'] <= 6 else 'long'
        tl = meta['text'].lower()
        rel_words = rel_words_ru if lang == 'ru' else rel_words_en
        complexity_tag = 'complex' if any(w in tl for w in rel_words) else 'simple'
        sentence_types.append({'length': length_tag, 'complexity': complexity_tag})

    # ── Tokenize ──────────────────────────────────────────────────────
    print(f"  Tokenizing {len(valid_sentences)} sentences (max_length={max_length})...")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    inputs = tokenizer(
        valid_sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return inputs, labels, valid_metadata, sentence_types


# ── Quick test loader (no stanza needed) ──────────────────────────────────────

def get_quick_test_data(tokenizer_name='gpt2-medium'):
    """
    Minimal dataset for dry-run testing (no stanza needed).
    Returns the same format as load_agreement_dataset.
    """
    sentences = [
        "The cat meows loudly.",
        "The cats meow loudly.",
        "The dog barks outside.",
        "The dogs bark outside.",
        "The child plays happily.",
        "The children play happily.",
        "A bird sings nearby.",
        "The birds sing nearby.",
    ]
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    dep_metadata = [
        {
            'text':               s,
            'subject_idx':        1,
            'verb_idx':           2,
            'subject_char_start': None,
            'verb_char_start':    None,
            # GPT-2 tokenises these simple sentences 1-to-1 per word,
            # so word index == token index for the test sentences.
            'subject_token_idx':  1,
            'verb_token_idx':     2,
            'is_plural':          int(l),
            'sentence_length':    len(s.split()),
        }
        for s, l in zip(sentences, labels)
    ]
    sentence_types = [
        {'length': 'short', 'complexity': 'simple'}
        for _ in sentences
    ]

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    inputs = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=32,
    )

    return inputs, labels, dep_metadata, sentence_types
