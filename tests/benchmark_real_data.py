#!/usr/bin/env python3
"""
Full-scale benchmark of Akana Grammatical Error Detection (GED) and Correction (GEC)
on the official GGLab/GECTurk datasets:
1. In-Domain Test Set: 20,769 sentences (230,571 tokens, 25 error categories)
2. Out-of-Domain Movie Reviews: 300 manually annotated real human sentences
"""

import os
import json
import urllib.request
import io
import time
import pyarrow.parquet as pq
import akana

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache_gecturk")
os.makedirs(CACHE_DIR, exist_ok=True)

def download_dataset(split_filename):
    local_path = os.path.join(CACHE_DIR, os.path.basename(split_filename))
    if os.path.exists(local_path):
        print(f"Loading cached dataset from: {local_path}")
        table = pq.read_table(local_path)
        return table.to_pylist()
        
    url = f"https://huggingface.co/datasets/GGLab/GECTurk/resolve/main/{split_filename}"
    print(f"Downloading dataset from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        table = pq.read_table(io.BytesIO(data))
        return table.to_pylist()

def calculate_f_beta(p, r, beta=0.5):
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * (p * r) / (b2 * p + r)

def evaluate_split(pylist, split_name="In-Domain Test Set", limit=None):
    if limit:
        pylist = pylist[:limit]
        
    print("\n" + "=" * 80)
    print(f" EVALUATING AKANA ON REAL DATASET: {split_name} ({len(pylist):,} sentences)")
    print("=" * 80)
    
    checker = akana.GrammarChecker()
    
    total_tokens = 0
    gold_error_tokens = 0
    pred_error_tokens = 0
    tp_tokens = 0
    fp_tokens = 0
    fn_tokens = 0
    tn_tokens = 0
    
    tp_sents = 0
    fp_sents = 0
    fn_sents = 0
    tn_sents = 0
    
    start_time = time.perf_counter()
    
    for row_idx, row in enumerate(pylist):
        tokens = row["tokens"]
        labels = row["labels"]
        sentence_text = " ".join(tokens)
        
        # Fast Rust JSON call
        res_json = checker.check_json(sentence_text)
        res_dict = json.loads(res_json)
        findings = res_dict.get("findings", [])
        
        has_gold_error = any(l > 0 for l in labels)
        has_pred_error = len(findings) > 0
        
        if has_gold_error and has_pred_error:
            tp_sents += 1
        elif not has_gold_error and has_pred_error:
            fp_sents += 1
        elif has_gold_error and not has_pred_error:
            fn_sents += 1
        else:
            tn_sents += 1
            
        token_has_finding = [False] * len(tokens)
        curr_char = 0
        for t_idx, tok in enumerate(tokens):
            t_start = sentence_text.find(tok, curr_char)
            if t_start == -1:
                t_start = curr_char
            t_end = t_start + len(tok)
            curr_char = t_end
            
            for f in findings:
                f_start = f.get("start_offset", 0)
                f_end = f.get("end_offset", 0)
                if not (f_end <= t_start or f_start >= t_end):
                    token_has_finding[t_idx] = True
                    break
                    
            is_gold_err = (labels[t_idx] > 0)
            is_pred_err = token_has_finding[t_idx]
            
            total_tokens += 1
            if is_gold_err:
                gold_error_tokens += 1
            if is_pred_err:
                pred_error_tokens += 1
                
            if is_gold_err and is_pred_err:
                tp_tokens += 1
            elif not is_gold_err and is_pred_err:
                fp_tokens += 1
            elif is_gold_err and not is_pred_err:
                fn_tokens += 1
            else:
                tn_tokens += 1
                
        if (row_idx + 1) % 5000 == 0 or (row_idx + 1) == len(pylist):
            now = time.perf_counter()
            cur_speed = (row_idx + 1) / (now - start_time)
            print(f"  Processed {row_idx + 1:,}/{len(pylist):,} sentences ({cur_speed:,.1f} sent/s)...")
                
    elapsed = time.perf_counter() - start_time
    throughput = len(pylist) / elapsed
    latency_us = (elapsed / len(pylist)) * 1_000_000
    
    # Token-level metrics
    p_tok = tp_tokens / (tp_tokens + fp_tokens) if (tp_tokens + fp_tokens) > 0 else 0.0
    r_tok = tp_tokens / (tp_tokens + fn_tokens) if (tp_tokens + fn_tokens) > 0 else 0.0
    f1_tok = calculate_f_beta(p_tok, r_tok, beta=1.0)
    f05_tok = calculate_f_beta(p_tok, r_tok, beta=0.5)
    
    # Sentence-level metrics
    p_sent = tp_sents / (tp_sents + fp_sents) if (tp_sents + fp_sents) > 0 else 0.0
    r_sent = tp_sents / (tp_sents + fn_sents) if (tp_sents + fn_sents) > 0 else 0.0
    f1_sent = calculate_f_beta(p_sent, r_sent, beta=1.0)
    f05_sent = calculate_f_beta(p_sent, r_sent, beta=0.5)
    
    print(f"\n⚡ SPEED & THROUGHPUT:")
    print(f"  • Total Sentences Processed: {len(pylist):,}")
    print(f"  • Total Tokens Analyzed:     {total_tokens:,}")
    print(f"  • Total Processing Time:     {elapsed:.3f} s")
    print(f"  • Throughput:                {throughput:,.1f} sentences/sec ({total_tokens/elapsed:,.1f} tokens/sec)")
    print(f"  • Avg Latency:               {latency_us:.2f} µs/sentence ({latency_us/1000:.4f} ms)")
    
    print(f"\n🎯 TOKEN-LEVEL ERROR DETECTION (GED):")
    print(f"  • Gold Error Tokens:         {gold_error_tokens:,}")
    print(f"  • Predicted Error Tokens:    {pred_error_tokens:,}")
    print(f"  • True Positives (TP):       {tp_tokens:,}")
    print(f"  • False Positives (FP):      {fp_tokens:,}")
    print(f"  • False Negatives (FN):      {fn_tokens:,}")
    print(f"  • Precision (P):             {p_tok * 100:.2f}%")
    print(f"  • Recall (R):                {r_tok * 100:.2f}%")
    print(f"  • F1 Score:                  {f1_tok * 100:.2f}%")
    print(f"  • F0.5 Score:                {f05_tok * 100:.2f}%")
    
    print(f"\n📊 SENTENCE-LEVEL ERROR DETECTION:")
    print(f"  • Precision (P):             {p_sent * 100:.2f}%")
    print(f"  • Recall (R):                {r_sent * 100:.2f}%")
    print(f"  • F0.5 Score:                {f05_sent * 100:.2f}%")
    
    return {
        "split": split_name,
        "sentences": len(pylist),
        "tokens": total_tokens,
        "throughput": throughput,
        "latency_ms": latency_us / 1000,
        "p_tok": p_tok,
        "r_tok": r_tok,
        "f1_tok": f1_tok,
        "f05_tok": f05_tok,
        "p_sent": p_sent,
        "r_sent": r_sent,
        "f05_sent": f05_sent,
    }

if __name__ == "__main__":
    # 1. Evaluate on Real Human Movie Reviews (300 sentences)
    movie_reviews_data = download_dataset("data/movie_reviews-00000-of-00001-b0322cba85108ee9.parquet")
    movie_results = evaluate_split(movie_reviews_data, split_name="Human Movie Reviews (Out-of-Domain Test Set)")
    
    # 2. Evaluate on Full In-Domain Test Set (20,769 sentences)
    test_data = download_dataset("data/test-00000-of-00001-82bf0acf5b396a77.parquet")
    test_results = evaluate_split(test_data, split_name="Full GECTurk In-Domain Test Set (20,769 Sentences)")
    
    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Benchmark Dataset':<35} {'Throughput':<16} {'Latency':<12} {'Precision':<10} {'Recall':<10} {'F0.5':<10}")
    print("-" * 95)
    print(f"{'Human Movie Reviews (300 sents)':<35} {movie_results['throughput']:,.0f} sent/s   {movie_results['latency_ms']:.3f} ms     {movie_results['p_tok']*100:.1f}%      {movie_results['r_tok']*100:.1f}%     {movie_results['f05_tok']*100:.1f}%")
    print(f"{'Full In-Domain Test (20,769 sents)':<35} {test_results['throughput']:,.0f} sent/s   {test_results['latency_ms']:.3f} ms     {test_results['p_tok']*100:.1f}%      {test_results['r_tok']*100:.1f}%     {test_results['f05_tok']*100:.1f}%")
