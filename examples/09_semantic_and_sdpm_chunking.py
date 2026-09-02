#!/usr/bin/env python3
"""Example 09: Semantic, Sentence, and SDPM Text Chunking with Akana.

Demonstrates high-performance text chunking strategies for Turkish RAG pipelines:
1. SemanticChunker: Splits documents at semantic topic shifts based on TurboQuant embeddings.
2. SentenceChunker: Fast sentence-boundary chunking with token constraints and overlap.
3. SDPMChunker (Semantic Double-Pass Merge): Two-pass clustering for optimal chunk density.
"""

import json
import akana

SAMPLE_TEXT = """
Kuantum mekaniği, atomik ve atom altı parçacıkların davranışlarını inceleyen temel fizik dalıdır. \
Dalga-parçacık ikiliği, süperpozisyon ve kuantum dolanıklığı gibi kavramlar teorinin bel kemiğini oluşturur. \
Son yıllarda geliştirilen kuantum bilgisayarlar, karmaşık kriptografi ve moleküler simülasyon hesaplamalarında çığır açmaktadır.

Fenerbahçe futbol takımı, Süper Lig derbisinde dün akşam üstün bir mücadele sergileyerek kritik üç puanın sahibi oldu. \
Teknik direktörün ikinci yarıda yaptığı taktiksel hamleler ve kanat organizasyonları taraftarlardan tam not aldı. \
Haftaya oynanacak deplasman karşılaşması şampiyonluk yarışında belirleyici olacaktır.

Geleneksel Türk mutfağı, zengin baharat çeşitliliği ve yüzlerce yıllık saray kültürüyle dünya gastronomisinde seçkin bir yere sahiptir. \
Kayseri mantısı, Adana kebabı ve taş fırında pişen Karadeniz pidesi coğrafi işaret tesciliyle koruma altındadır. \
Tatlı kültüründe ise fıstıklı Gaziantep baklavası ve fırın sütlaç vazgeçilmez lezzetlerdendir.
"""

def main():
    print("=" * 70)
    print(" ⚡ AKANA TURKISH TEXT CHUNKING SUITE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. SemanticChunker (Embedding-based topic split)
    # ---------------------------------------------------------
    print("\n[1] SemanticChunker (Percentile Mode, Threshold=0.75):")
    semantic_chunker = akana.SemanticChunker(
        chunk_size=512,
        threshold_mode="percentile",  # "percentile", "similarity", "stdev", "iqr", "auto"
        threshold_value=0.75,
        min_chunk_size=15,
        use_smoothing=True,
    )
    semantic_chunks = semantic_chunker(SAMPLE_TEXT)
    print(f"Produced {len(semantic_chunks)} semantic topic chunks:")
    for idx, c in enumerate(semantic_chunks, 1):
        print(f"\n  Chunk #{idx} [{c.token_count} tokens | chars {c.start_index}:{c.end_index}]:")
        print(f"  \"{c.text[:90]}...\"")
        print(f"  Sentences contained: {len(c.sentences)}")

    # ---------------------------------------------------------
    # 2. SentenceChunker (Sentence-boundary-aware with overlap)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[2] SentenceChunker (Chunk Size=40 tokens, Overlap=10 tokens):")
    sentence_chunker = akana.SentenceChunker(
        chunk_size=40,
        chunk_overlap=10,
        min_sentences_per_chunk=1,
    )
    sentence_chunks = sentence_chunker(SAMPLE_TEXT)
    print(f"Produced {len(sentence_chunks)} sentence chunks:")
    for idx, c in enumerate(sentence_chunks, 1):
        print(f"  Chunk #{idx} ({c.token_count} tokens, chars {c.start_index}-{c.end_index}): {c.text[:60]}...")

    # ---------------------------------------------------------
    # 3. SDPMChunker (Semantic Double-Pass Merge)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[3] SDPMChunker (Two-Pass Merge, Merge Threshold=0.60):")
    sdpm_chunker = akana.SDPMChunker(
        chunk_size=512,
        threshold_mode="percentile",
        threshold_value=0.75,
        merge_threshold=0.60,
    )
    sdpm_chunks = sdpm_chunker(SAMPLE_TEXT)
    print(f"Produced {len(sdpm_chunks)} merged chunks:")
    for idx, c in enumerate(sdpm_chunks, 1):
        print(f"  SDPM Chunk #{idx} ({c.token_count} tokens): {c.text[:75]}...")

    # ---------------------------------------------------------
    # 4. JSON Export & Dict Conversion
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[4] Chunk Object Dict & JSON Serialization:")
    first_dict = semantic_chunks[0].to_dict()
    print("Dict representation keys:", list(first_dict.keys()))
    print("JSON snippet:\n", json.dumps(first_dict, ensure_ascii=False, indent=2)[:300], "...\n}")

if __name__ == "__main__":
    main()
