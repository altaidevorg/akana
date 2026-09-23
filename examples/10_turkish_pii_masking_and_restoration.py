#!/usr/bin/env python3
"""Example 10: Turkish PII Masking and Bi-Directional Restoration with Akana.

Demonstrates high-performance, KVKK-aligned Turkish PII anonymization and restoration:
1. Structured Placeholder Masking ({{AD_1}}, {{TCKN_1}}, {{IBAN_1}}) with full mapping.
2. Lossless De-anonymization / Restoration with Turkish vowel harmony preservation.
3. Realistic Turkish Synthetic Surrogates for fluent downstream processing.
4. Specific Private Date Detection (appointments, billing, birth dates vs public calendar dates).
5. Private URLs with auth tokens and corporate support email preservation.
6. Secrets, Passwords, OTP codes, and API keys detection.
7. Model2Vec 256-dim embedding disambiguation for polysemous words (e.g. Deniz, Barış).
"""

import akana


def main():
    print("=" * 75)
    print(" [AKANA] TURKISH PII MASKING & RESTORATION SUITE (KVKK ALIGNED)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Structured Placeholder Masking & Dictionary Mapping
    # -------------------------------------------------------------------------
    print("\n--- 1. Structured Placeholder Masking & Lossless Restoration ---")
    prompt1 = (
        "Müşterimiz Ahmet Yılmaz, 10000000146 nolu TCKN ve 0532 123 45 67 numaralı "
        "telefonuyla başvuruda bulunmuş olup TR33 0006 1005 1978 6457 8413 26 nolu "
        "IBAN hesabına 15.000 TL kredi talep etmektedir."
    )
    result1 = akana.pii_mask(prompt1, mode="placeholder")

    print("Original Text:")
    print(f"  {prompt1}")
    print("\nMasked Output:")
    print(f"  {result1['masked_text']}")
    print("\nGenerated Mapping Dictionary:")
    for placeholder, original in result1["mapping"].items():
        print(f"  {placeholder:<14} -> '{original}'")

    # Downstream restoration using only the mapping dictionary
    restored1 = akana.pii_restore(result1["masked_text"], result1["mapping"])
    print(
        f"\nRestoration Check: {'PASSED (Lossless)' if restored1 == prompt1 else 'FAILED'}"
    )

    # -------------------------------------------------------------------------
    # 2. Turkish Morphology & Vowel Harmony Preservation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("--- 2. Vowel Harmony Preservation on Inflected Suffixes ---")
    prompt2 = "Ödeme Ahmet Yılmaz'a aktarılmış ve Zeynep Kaya'nın hesabına geçmiştir."
    result2 = akana.pii_mask(prompt2, mode="placeholder")

    print(f"Original Text:  {prompt2}")
    print(f"Masked Text:    {result2['masked_text']}")

    # Simulate downstream text where placeholders carry case suffixes
    simulated_downstream = (
        "İşlem tamam: {{AD_1}}'a bilgilendirme SMS'i gönderildi, "
        "ayrıca {{AD_2}}'nın dosyası onaylandı."
    )
    restored2 = akana.pii_restore(simulated_downstream, result2["mapping"])
    print(f"Downstream Input: {simulated_downstream}")
    print(f"Harmonized Restoration:\n  {restored2}")

    # -------------------------------------------------------------------------
    # 3. Realistic Synthetic Surrogates Mode
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("--- 3. Realistic Turkish Synthetic Surrogates Mode ---")
    prompt3 = "Ali Kaya, 10000000146 nolu TCKN ile dün akşam Kadıköy şubesine geldi."
    result3 = akana.pii_mask(prompt3, mode="surrogate")

    print(f"Original Text:  {prompt3}")
    print(f"Synthetic Text: {result3['masked_text']}")
    print("Surrogate Mapping:")
    for surrogate, original in result3["mapping"].items():
        print(f"  '{surrogate}' -> '{original}'")

    restored3 = akana.pii_restore(result3["masked_text"], result3["mapping"])
    print(f"Restored:       {restored3}")

    # -------------------------------------------------------------------------
    # 4. Specific Private Dates vs Public Calendar Dates
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("--- 4. Specific Private Dates vs Public Calendar Dates ---")
    date_text = (
        "Randevu günüm 2 mayıs 2011 görünmesine rağmen fatura kesim tarihim: 01.10.2025 yazıyor. "
        "Ayrıca yılbaşı fırsatı kodu 31.12.2026 ye kadar geçerli ve 30 ağustos duyurusu "
        "30 ağustos 2026 tarihinde başlayacak."
    )
    date_result = akana.pii_mask(date_text, mode="tag")
    print("Input Text:")
    print(f"  {date_text}")
    print(
        "\nMasked Output (Notice private dates masked, public calendar dates preserved):"
    )
    print(f"  {date_result['masked_text']}")

    # -------------------------------------------------------------------------
    # 5. Private URLs and Corporate Support Email Preservation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("--- 5. Private URLs & Corporate Support Email Preservation ---")
    corp_text = (
        "Kişisel e-postam derin.demir@example.net üzerinden bildirim alamadım, "
        "destek@hotelgo.net ve info@sirket.com.tr adreslerine yazdım. "
        "Özel dosya linki https://portal.example.org/dosya/658033-bulutbugun linkindedir."
    )
    corp_result = akana.pii_mask(
        corp_text, mode="placeholder", preserve_corporate_emails=True
    )

    print("Input Text:")
    print(f"  {corp_text}")
    print("\nMasked Output (Corporate emails preserved, private URL and email masked):")
    print(f"  {corp_result['masked_text']}")
    print("\nExtracted Entities:")
    for e in corp_result["entities"]:
        print(f"  [{e['label']}] '{e['text']}' (Confidence: {e['confidence']:.2f})")

    # -------------------------------------------------------------------------
    # 6. Passwords, Secrets, OTP Codes & API Tokens
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("--- 6. Passwords, Secrets, OTPs & API Tokens ---")
    secret_text = (
        "Giriş yapamıyorum: şifrem Limon004_ ile hata alıyorum. "
        "Gelen sms doğrulama kodum 321165 ve mobil onay pinim 45606. "
        "OpenAI anahtarım sk-proj-12345678901234567890123456789012."
    )
    secret_result = akana.pii_mask(secret_text, mode="tag")
    print("Input Text:")
    print(f"  {secret_text}")
    print("\nMasked Output:")
    print(f"  {secret_result['masked_text']}")

    # -------------------------------------------------------------------------
    # 7. Embedding-Powered Polysemous Name Disambiguation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("--- 7. Model2Vec 256-Dim Embedding Disambiguation ---")
    sent1 = "Deniz dün ofise geç geldi ve raporu teslim etti."
    sent2 = "Yaz tatilinde deniz kenarında yürüyüş yaptık."

    res_emb1 = akana.pii_mask(sent1, mode="tag", use_embeddings=True)
    res_emb2 = akana.pii_mask(sent2, mode="tag", use_embeddings=True)

    print(f"Human context:  '{sent1}' -> '{res_emb1['masked_text']}'")
    print(f"Nature context: '{sent2}' -> '{res_emb2['masked_text']}'")

    print("\n" + "=" * 75)
    print(" Demo completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
