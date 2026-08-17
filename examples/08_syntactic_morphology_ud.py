"""
Example 08: Turkish Syntactic Expressive Morphology (Google Research / FSMNLP 2019)
Demonstrates:
1. Two-level Inflectional Group (IG) hierarchy.
2. Zero-derivation elimination (Cross-categorized lexical entries).
3. Universal Dependencies (UD) category-value feature sets.
4. Side-by-side comparison between Zemberek flat parsing vs Google-style Syntactic parsing.
"""

import akana

def compare_morphology(word: str):
    print(f"\n{'='*75}")
    print(f" MORPHOLOGICAL COMPARISON FOR: '{word}'")
    print(f"{'='*75}")

    # 1. Traditional Zemberek / Oflazer Flat Parse (Broad 93k Lexicon)
    z_analyzer = akana.Morphology()
    z_parses = z_analyzer.analyze(word)

    print("\n[1] Traditional Zemberek / Oflazer Flat Representation:")
    for i, p in enumerate(z_parses, 1):
        print(f"    {i}. {p['formatted']} (Lemma: {p['lemma']}, POS: {p['primary_pos']})")
        print(f"       Morpheme Sequence: {' + '.join(p['morphemes'])}")

    # 2. Google Expressive / Syntactic Inflectional Group Parse (FSMNLP 2019)
    g_parses = akana.syntactic_analyze(word)

    print("\n[2] Google-Style Syntactic Expressive Inflectional Groups (IG):")
    for i, p in enumerate(g_parses, 1):
        print(f"    {i}. {p.formatted}")
        print(f"       Root: {p.root} [{p.root_pos}], Proper: {p.is_proper}")
        for j, ig in enumerate(p.inflectional_groups):
            deriv_info = f" (Derivation: {ig.derivation})" if ig.derivation else " (Root Tier)"
            print(f"       • IG_{j}: [{ig.pos}]{deriv_info} -> {ig.features}")

def main():
    print("=" * 75)
    print(" AKANA TURKISH SYNTACTIC MORPHOLOGY & INFLECTIONAL GROUPS")
    print("=" * 75)

    test_words = [
        "geldiğimizde",
        "evlerimizde",
        "güzel",
        "öğretmenlerimizden",
    ]

    for w in test_words:
        compare_morphology(w)

    print(f"\n{'='*75}")
    print(" DEMONSTRATION FINISHED SUCCESSFULLY!")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
