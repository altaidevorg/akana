use akana_core::pii::*;

#[test]
fn test_checksum_validators_integration() {
    // Valid TCKN
    assert!(validate_tckn("10000000146"));
    assert!(validate_tckn("51980838978"));
    assert!(!validate_tckn("10203040506"));
    assert!(!validate_tckn("11111111110"));

    // Valid VKN
    assert!(validate_vkn("1234567890"));
    assert!(!validate_vkn("1234567891"));

    // Valid TR IBAN
    assert!(validate_iban("TR330006100519786457841326"));
    assert!(validate_iban("TR33 0006 1005 1978 6457 8413 26"));
    assert!(!validate_iban("TR330006100519786457841327"));

    // Credit Cards
    assert_eq!(
        validate_credit_card("4111 1111 1111 1111"),
        Some(CardBrand::Visa)
    );
    assert_eq!(
        validate_credit_card("9792 0000 0000 0003"),
        Some(CardBrand::Troy)
    );
    assert_eq!(validate_credit_card("4111 1111 1111 1112"), None);

    // Vehicle Plates
    assert!(validate_plate("34 ABC 123"));
    assert!(validate_plate("06 A 1234"));
    assert!(validate_plate("35 AB 1234"));
    assert!(!validate_plate("82 ABC 123"));

    // VIN
    assert!(validate_vin("1HGCM82633A004352"));
    assert!(!validate_vin("1HGCM82633A004353"));
}

#[test]
fn test_complex_banking_prompt_masking_and_restoration() {
    let engine = TurkishPiiEngine::new();
    let prompt = "Müşteri Ahmet Yılmaz (TC: 10000000146, Tel: 0532 123 45 67), TR330006100519786457841326 no'lu IBAN hesabına 50.000 TL transfer talep etti.";

    let res = engine.mask(prompt, PiiMode::Placeholder);

    // Masked assertions
    assert!(res.masked_text.contains("{{AD_1}}"));
    assert!(res.masked_text.contains("{{TCKN_1}}"));
    assert!(res.masked_text.contains("{{TEL_1}}"));
    assert!(res.masked_text.contains("{{IBAN_1}}"));
    assert!(!res.masked_text.contains("Ahmet Yılmaz"));
    assert!(!res.masked_text.contains("10000000146"));
    assert!(!res.masked_text.contains("0532 123 45 67"));
    assert!(!res.masked_text.contains("TR330006100519786457841326"));

    // Simulate LLM response
    let llm_reply = "{{AD_1}} adına kayıtlı {{IBAN_1}} hesabına transfer başarıyla gerçekleşti. Bilgi SMS'i {{TEL_1}} numarasına gönderildi.";
    let restored = engine.restore_response(llm_reply, &res.vault);

    assert_eq!(
        restored,
        "Ahmet Yılmaz adına kayıtlı TR330006100519786457841326 hesabına transfer başarıyla gerçekleşti. Bilgi SMS'i 0532 123 45 67 numarasına gönderildi."
    );
}

#[test]
fn test_synthetic_surrogate_mode() {
    let engine = TurkishPiiEngine::new();
    let prompt = "Müşteri Mehmet Kaya telefon numarası 0532 111 22 33 olan hesabı sorguladı.";

    let res = engine.mask(prompt, PiiMode::SyntheticSurrogate);

    // Surrogate should replace with fake Turkish name/phone
    assert!(!res.masked_text.contains("Mehmet Kaya"));
    assert!(!res.masked_text.contains("0532 111 22 33"));
    assert!(res.masked_text.contains("Can Demir"));

    // Restoration
    let llm_reply = "Can Demir isimli müşterinin sorgulama işlemi tamamlandı.";
    let restored = engine.restore_response(llm_reply, &res.vault);
    assert_eq!(
        restored,
        "Mehmet Kaya isimli müşterinin sorgulama işlemi tamamlandı."
    );
}

#[test]
fn test_morphological_vowel_harmony_restoration() {
    let engine = TurkishPiiEngine::new();

    // 1. Murat (back vowel 'a')
    let prompt1 = "Murat'a bilgi verildi.";
    let res1 = engine.mask(prompt1, PiiMode::Placeholder);
    assert!(res1.masked_text.contains("{{AD_1}}'a"));

    let llm_resp1 = "{{AD_1}}'a işlem onaylandı.";
    let restored1 = engine.restore_response(llm_resp1, &res1.vault);
    assert_eq!(restored1, "Murat'a işlem onaylandı.");

    // 2. Ali (front vowel 'i')
    let prompt2 = "Ali'ye e-posta iletildi.";
    let res2 = engine.mask(prompt2, PiiMode::Placeholder);
    assert!(res2.masked_text.contains("{{AD_1}}'ye"));

    let llm_resp2 = "{{AD_1}}'ye e-posta gönderildi.";
    let restored2 = engine.restore_response(llm_resp2, &res2.vault);
    assert_eq!(restored2, "Ali'ye e-posta gönderildi.");
}

#[test]
fn test_polysemy_resistance() {
    let engine = TurkishPiiEngine::new();

    // "Deniz kenarı" should not mask
    let text1 = "Deniz kenarında yürüyüş yaptık.";
    let res1 = engine.mask(text1, PiiMode::Tag);
    assert_eq!(res1.masked_text, text1);

    // "Toprak kayması" should not mask
    let text2 = "Yoğun yağış sebebiyle toprak kayması meydana geldi.";
    let res2 = engine.mask(text2, PiiMode::Tag);
    assert_eq!(res2.masked_text, text2);

    // "Sayın Deniz Hanım" SHOULD mask
    let text3 = "Sayın Deniz Hanım toplantıya katıldı.";
    let res3 = engine.mask(text3, PiiMode::Tag);
    assert!(res3.masked_text.contains("[AD]"));
}

#[test]
fn test_address_and_sensitive_categories() {
    let engine = TurkishPiiEngine::new();
    let text = "Hasta kan grubu 0 Rh+ olup diyabet tedavisi görmektedir. Adres: Atatürk Mahallesi, Cumhuriyet Caddesi No: 12 Kadıköy / İstanbul.";

    let res = engine.mask(text, PiiMode::Tag);

    assert!(res.masked_text.contains("[KAN_GRUBU]"));
    assert!(res.masked_text.contains("[SAGLIK]"));
    assert!(res.masked_text.contains("[ADRES]"));
}
