"""
Example 02: Spell Checking with StringZilla SIMD, De-asciification & Normalization
"""

import akana

def main():
    print("=" * 60)
    print("      AKANA: NORMALIZATION & SIMD SPELL CHECKING")
    print("=" * 60)

    # 1. StringZilla SIMD-Accelerated Spell Checker
    print("\n--- 1. StringZilla SIMD Spell Checking ---")
    spell = akana.SpellChecker()

    test_words = ["kitap", "ktap", "kitaplra", "türkiye", "evler", "evlerl"]
    for w in test_words:
        is_ok = spell.is_correct(w)
        status = "CORRECT" if is_ok else "TYPO"
        print(f"Word: '{w:<10}' -> {status}")
        if not is_ok:
            suggs = spell.suggest(w, max_distance=2, max_suggestions=3)
            print(f"  Suggestions: {[s['word'] for s in suggs]}")

    # 2. De-asciification (Restoring Turkish diacritics)
    print("\n--- 2. Turkish De-asciification ---")
    ascii_texts = [
        "turkce nlp cok hizli calisiyor",
        "ogrenci kutuphanede kitap okuyor ve caliskan",
        "agaclar ve cicekler ilkbaharda cok guzel aciyor"
    ]
    for text in ascii_texts:
        restored = akana.deasciify(text)
        print(f"  Input:    {text}")
        print(f"  Restored: {restored}\n")

    # 3. Informal & Chat Contractions Normalization
    print("--- 3. Informal Spoken Contractions Normalization ---")
    chat_messages = [
        "nooldu ya neden gelmiyon",
        "yarin ben de yapcam",
        "slm nbr nasılsın tşk",
        "çooookkk güzel bir film"
    ]
    for msg in chat_messages:
        normalized = akana.normalize_informal(msg)
        print(f"  Informal:   {msg}")
        print(f"  Normalized: {normalized}\n")

if __name__ == "__main__":
    main()
