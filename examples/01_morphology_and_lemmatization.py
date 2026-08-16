"""
Example 01: Morphological Analysis and Generation with Akana
"""

import akana

def main():
    print("=" * 60)
    print("          AKANA: MORPHOLOGY & LEMMATIZATION DEMO")
    print("=" * 60)

    morph = akana.Morphology()

    # 1. Morphological Analysis
    words_to_analyze = [
        "kitaplarımızda",
        "evlerimizde",
        "burnum",
        "hakkım",
        "geliyorum",
        "gideceğim",
        "çalışkan"
    ]

    print("\n--- 1. Morphological Analysis ---")
    for word in words_to_analyze:
        parses = morph.analyze(word)
        print(f"\nWord: '{word}' (Total valid parses: {len(parses)})")
        for i, p in enumerate(parses, 1):
            print(f"  [{i}] Root: {p['root']:<10} Lemma: {p['lemma']:<10} POS: {p['primary_pos']:<6} Morphemes: {p['morphemes']}")
            print(f"      Formatted: {p['formatted']}")

    # 2. Morphological Generation (Surface synthesis from tags)
    print("\n--- 2. Morphological Surface Generation ---")
    generation_tasks = [
        ("kitap", ["Noun", "A3sg", "P1sg", "Dat"]),
        ("burun", ["Noun", "A3sg", "P1sg"]),
        ("hak", ["Noun", "A3sg", "P1sg"]),
        ("gel", ["Verb", "Prog", "A1sg"]),
        ("git", ["Verb", "Fut", "A1sg"]),
    ]

    for lemma, tags in generation_tasks:
        surface = morph.generate(lemma, tags)
        print(f"  Lemma '{lemma}' + Tags {tags} -> Surface: '{surface}'")

if __name__ == "__main__":
    main()
