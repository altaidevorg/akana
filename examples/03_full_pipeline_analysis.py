"""
Example 03: End-to-End Turkish Document Pipeline Analysis
"""

import akana

def main():
    print("=" * 60)
    print("        AKANA: END-TO-END DOCUMENT NLP PIPELINE")
    print("=" * 60)

    sample_article = (
        "Ak Ana, Türk ve Altay mitolojisinde deniz tanrıçasıdır. "
        "Prof. Dr. Ahmet Bey konferansta konuştu! "
        "Ali yeni bir kitap aldı ve kütüphanede okudu."
    )

    print(f"\nProcessing Text:\n\"\"\"\n{sample_article}\n\"\"\"\n")

    doc = akana.analyze(sample_article)
    print(f"Detected {len(doc.sentences)} sentences:\n")

    for i, sent in enumerate(doc.sentences, 1):
        print(f"[Sentence {i}]: \"{sent.text}\"")
        print(f"  Tokens ({len(sent.tokens)}): {sent.tokens}")
        print("  Morphological Disambiguation:")
        for p in sent.parses:
            tags = p.get('morpheme_tags') or p.get('morphemes', [])
            print(f"    - {p['surface']:<14} -> Lemma: {p['lemma']:<12} POS: {p['primary_pos']:<6} Tags: {tags}")
        print()

if __name__ == "__main__":
    main()
