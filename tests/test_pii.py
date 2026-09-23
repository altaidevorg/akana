import akana


def test_mask_and_restore_with_dictionary_mapping():
    prompt = "Müşterimiz Ahmet Yılmaz, 10000000146 nolu TCKN ve 0532 123 45 67 numaralı telefonuna sahiptir."
    result = akana.pii_mask(prompt, mode="placeholder")

    # 1. Output schema validation
    assert "masked_text" in result
    assert "mapping" in result
    assert "detailed_mapping" in result
    assert "entities" in result
    assert "vault" in result

    # 2. Check mapping dictionary
    mapping = result["mapping"]
    assert isinstance(mapping, dict)
    assert len(mapping) >= 3

    masked = result["masked_text"]
    assert "10000000146" not in masked
    assert "0532 123 45 67" not in masked
    assert "Ahmet Yılmaz" not in masked

    # 3. Downstream restoration using ONLY the returned mapping dictionary
    llm_response = f"Bilgi: {masked} onaylanmıştır."
    restored = akana.pii_restore(llm_response, mapping)

    assert "10000000146" in restored
    assert "0532 123 45 67" in restored
    assert "Ahmet Yılmaz" in restored
    assert restored == f"Bilgi: {prompt} onaylanmıştır."


def test_vowel_harmony_with_mapping_restoration():
    prompt = "Ödeme Ahmet Yılmaz'a aktarılmış ve Zeynep Kaya'nın hesabına geçmiştir."
    result = akana.pii_mask(prompt, mode="placeholder")
    mapping = result["mapping"]

    # Restoring using raw dictionary preserves apostrophe suffixes
    restored = akana.pii_restore(result["masked_text"], mapping)
    assert restored == prompt


def test_synthetic_surrogate_mode():
    prompt = "Ali Kaya dün İstanbul'a gitti."
    result = akana.pii_mask(prompt, mode="surrogate")
    mapping = result["mapping"]

    masked = result["masked_text"]
    # Should replace Ali Kaya with a natural-sounding Turkish surrogate
    assert "Ali Kaya" not in masked
    assert len(mapping) >= 1

    # Restoring with the surrogate mapping restores Ali Kaya
    restored = akana.pii_restore(masked, mapping)
    assert "Ali Kaya" in restored


def test_engine_api_parity():
    engine = akana.TurkishPiiEngine()
    prompt = (
        "Sayın Caner Çetin, TR33 0006 1005 1978 6457 8413 26 nolu IBAN'ınız aktiftir."
    )

    res = engine.mask(prompt, mode="placeholder")
    assert "mapping" in res

    restored = engine.restore(res["masked_text"], res["mapping"])
    assert "TR33 0006 1005 1978 6457 8413 26" in restored
    assert "Caner Çetin" in restored


def test_advanced_enterprise_identifiers():
    text = (
        "IMEI: 356938035643803, MAC: 1F:FA:B2:B3:CF:1D, "
        "BTC: bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq, "
        "Poliçe: POL-99482103, Sözleşme: SOZLESME_NO: 2024-TR-8812, "
        "Şasi: şasi no 6Z9741GSRLHSCNA73, Motor: motor N66D336569, "
        "Kredi Notu: kredi notu 750"
    )
    res = akana.pii_mask(text)
    mapping = res["mapping"]

    # Ensure sensitive values are masked
    assert "356938035643803" not in res["masked_text"]
    assert "1F:FA:B2:B3:CF:1D" not in res["masked_text"]
    assert "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq" not in res["masked_text"]
    assert "POL-99482103" not in res["masked_text"]

    # Ensure restoration is 100% lossless
    restored = akana.pii_restore(res["masked_text"], mapping)
    assert restored == text


def test_pii_mask_and_pii_restore_naming():
    text = "Müşteri Ahmet Can tc 10000000146."
    res = akana.pii_mask(text)
    assert "masked_text" in res
    assert "10000000146" not in res["masked_text"]
    restored = akana.pii_restore(res["masked_text"], res["mapping"])
    assert "10000000146" in restored
    assert "Ahmet Can" in restored


def test_advanced_secrets_and_credentials():
    prompt = (
        "Giriş yapamıyorum: şifrem Limon004_ ile hata alıyorum. "
        "Mobil onay pinim 45606 ve gelen sms doğrulama kodum 321165. "
        "Ayrıca API anahtarım sk-proj-12345678901234567890123456789012."
    )
    res = akana.pii_mask(prompt)
    masked = res["masked_text"]
    mapping = res["mapping"]

    assert "Limon004_" not in masked
    assert "45606" not in masked
    assert "321165" not in masked
    assert "sk-proj-12345678901234567890123456789012" not in masked

    restored = akana.pii_restore(masked, mapping)
    assert restored == prompt


def test_private_dates_and_public_date_preservation():
    # 1. Positive private dates must be masked
    priv_prompt = "Randevu günüm 2 mayıs 2011 görünmesine rağmen fatura kesim tarihim: 01.10.2025 yazıyor."
    res1 = akana.pii_mask(priv_prompt, mode="tag")
    assert (
        "[OZEL_TARIH]" in res1["masked_text"] or "[DOGUM_TARIHI]" in res1["masked_text"]
    )
    assert "01.10.2025" not in res1["masked_text"]

    # 2. Public / campaign dates must NOT be masked
    pub_prompt = "yılbaşı fırsatı kodu 31.12.2026 ye kadar geçerli, 30 ağustos duyurusu 30 ağustos 2026 tarihinde başlayacak."
    res2 = akana.pii_mask(pub_prompt, mode="tag")
    assert res2["masked_text"] == pub_prompt


def test_private_urls_and_corporate_emails():
    text = (
        "hesabım derin.demir@example.net ile açılmış, satis@hotelgo.net adresine yazdım. "
        "kişisel dosya linki https://portal.example.org/dosya/658033-bulutbugun linkinde."
    )
    res = akana.pii_mask(text, mode="placeholder", preserve_corporate_emails=True)
    masked = res["masked_text"]
    mapping = res["mapping"]

    # Personal email is masked
    assert "derin.demir@example.net" not in masked
    assert "{{EMAIL_1}}" in masked

    # Corporate support email is preserved
    assert "satis@hotelgo.net" in masked

    # Private URL is masked
    assert "https://portal.example.org/dosya/658033-bulutbugun" not in masked
    assert "{{OZEL_URL_1}}" in masked

    # Lossless restoration
    restored = akana.pii_restore(masked, mapping)
    assert restored == text


def test_embedding_scorer_pii_mask():
    # Use embeddings for polysemous name disambiguation
    text1 = "Deniz dün ofise geç geldi ve raporu teslim etti."
    res1 = akana.pii_mask(text1, mode="tag", use_embeddings=True)
    assert "[AD]" in res1["masked_text"]

    text2 = "Yaz tatilinde deniz kenarında yürüyüş yaptık."
    res2 = akana.pii_mask(text2, mode="tag", use_embeddings=True)
    assert "[AD]" not in res2["masked_text"]
    assert res2["masked_text"] == text2
