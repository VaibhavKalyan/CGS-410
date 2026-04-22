"""
data_loader.py — Dataset loading and dependency parsing for English & Russian.

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
import random

import numpy as np
import torch

import stanza
from transformers import GPT2Tokenizer
from datasets import load_dataset


# ── Dependency parsing ────────────────────────────────────────────────────────

def _parse_dependencies(sentences, nlp):
    """
    Parse sentences with stanza and extract subject-verb dependency pairs.
    Returns list of dicts with 'subject_idx', 'verb_idx', 'is_plural' per sentence.
    """
    parsed = []
    for sent_text in sentences:
        dep_info = {
            'text': sent_text,
            'subject_idx': None,
            'verb_idx': None,
            'is_plural': None,
            'sentence_length': len(sent_text.split()),
        }
        try:
            doc = nlp(sent_text)
            for sent in doc.sentences:
                for word in sent.words:
                    # Accept both nsubj and nsubj:pass (passive subjects)
                    if word.deprel in ('nsubj', 'nsubj:pass'):
                        dep_info['subject_idx'] = word.id - 1   # 0-indexed
                        feats = word.feats if word.feats else ''
                        dep_info['is_plural'] = 1 if 'Number=Plur' in feats else 0
                        dep_info['verb_idx'] = word.head - 1    # 0-indexed
                        break
                if dep_info['subject_idx'] is not None:
                    break
        except Exception as e:
            # Silently skip sentences that crash stanza
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
                # Newer datasets versions need trust_remote_code
                ds = load_dataset(
                    "blimp", subset,
                    split="train",
                    trust_remote_code=True,
                )
            except TypeError:
                # Older datasets version without trust_remote_code arg
                ds = load_dataset("blimp", subset, split="train")
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

    # ── Stanza pipeline ───────────────────────────────────────────────
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
        use_gpu=torch.cuda.is_available(),   # use GPU on Colab if available
    )

    # ── Parse dependencies ────────────────────────────────────────────
    print(f"  Parsing {len(sentences)} sentences...")
    dep_metadata = _parse_dependencies(sentences, nlp)

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
            'text': s,
            'subject_idx': 1,
            'verb_idx': 2,
            'is_plural': int(l),
            'sentence_length': len(s.split()),
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
