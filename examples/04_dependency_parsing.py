"""
Example 04: Syntactic Dependency Parsing with Universal Dependencies & CoNLL-U
"""

import akana

def main():
    print("=" * 60)
    print("      AKANA: SYNTACTIC DEPENDENCY PARSING (CoNLL-U)")
    print("=" * 60)

    parser = akana.DependencyParser()

    sample_sentences = [
        ["Ali", "güzel", "kitabı", "okudu"],
        ["Öğretmen", "öğrencilere", "yeni", "bir", "konu", "anlattı"],
        ["Türkiye", "büyük", "ve", "güzel", "bir", "ülkedir"]
    ]

    for i, tokens in enumerate(sample_sentences, 1):
        print(f"\n[Sentence {i}]: {' '.join(tokens)}")
        conllu_output = parser.parse_conllu(tokens)
        print("-" * 50)
        print(f"{'ID':<4} {'FORM':<12} {'LEMMA':<12} {'UPOS':<8} {'HEAD':<6} {'DEPREL':<10}")
        print("-" * 50)
        for line in conllu_output.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 8:
                print(f"{parts[0]:<4} {parts[1]:<12} {parts[2]:<12} {parts[3]:<8} {parts[6]:<6} {parts[7]:<10}")

if __name__ == "__main__":
    main()
