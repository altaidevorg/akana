//! Tests of the Turkish AI-text detector (`akana_core::aidetect`): cleaning, content
//! features, detection, explanation and parity with the Python reference.

/// Content-channel cleaning and format signals.
mod cleaning {
    use akana_core::aidetect::preprocess::{
        drop_chatbot_opener, mask, match_chatbot_opener, normalize, preprocess, preprocess_batch,
        strip_format, PreprocessOptions, CHATBOT_OPENERS,
    };

    fn all() -> PreprocessOptions {
        PreprocessOptions::default()
    }

    // ---------------------------------------------------------------- normalize

    #[test]
    fn normalize_keeps_turkish_case_and_multibyte_letters() {
        let s = "İSTANBUL'DA ılık bir gün; Iğdır ve ışık. ŞÇĞÜÖ şçğüö";
        assert_eq!(normalize(s), s);
    }

    #[test]
    fn normalize_typography() {
        assert_eq!(
            normalize("“Merhaba” dedi—sonra ‘tamam’… «evet» 1990–2000 a -- b"),
            "\"Merhaba\" dedi-sonra 'tamam'... \"evet\" 1990-2000 a - b"
        );
        // Curly apostrophe and acute accent used as apostrophes.
        assert_eq!(normalize("Türkiye’nin İzmir´de"), "Türkiye'nin İzmir'de");
    }

    #[test]
    fn normalize_removes_invisibles_and_nfkc() {
        let s =
        "\u{FEFF}Bu\u{200B}gün\u{00A0}hava\u{202F}çok\u{00AD} güzel\u{200D}\u{200C}. ﬁlm ＡＢＣ";
        assert_eq!(normalize(s), "Bugün hava çok güzel. film ABC");
    }

    #[test]
    fn normalize_whitespace_and_paragraphs() {
        let s = "  Birinci   satır\t burada.\r\nİkinci satır.\n\n\n   \n\nYeni paragraf.  \n";
        assert_eq!(
            normalize(s),
            "Birinci satır burada.\nİkinci satır.\n\nYeni paragraf."
        );
    }

    // ---------------------------------------------------------------- strip_format

    #[test]
    fn strip_nested_markdown() {
        let raw = "## **Önemli** Başlık\n\n\
               - **Kalın** ve *eğik* madde\n  \
               - İç içe _vurgulu_ ***ikisi*** madde\n\
               1. Numaralı [bağlantı](https://ornek.com) ve `kod`\n\
               > Alıntı ~~silik~~ satır\n\n\
               ---\n\
               Düz metin snake_case ve 2*3*4 kalır.";
        assert_eq!(
            strip_format(raw),
            "Önemli Başlık\n\n\
         Kalın ve eğik madde\n\
         İç içe vurgulu ikisi madde\n\
         Numaralı bağlantı ve kod\n\
         Alıntı silik satır\n\n\
         Düz metin snake_case ve 2*3*4 kalır."
        );
    }

    #[test]
    fn strip_table_joins_cells_into_sentences() {
        let raw = "Tablo:\n\n| Şehir | Nüfus |\n|:------|------:|\n| **Ankara** | 5,6 milyon |\n| İzmir | 4,4 milyon |\n\nSon.";
        assert_eq!(
            strip_format(raw),
            "Tablo:\n\nŞehir, Nüfus.\nAnkara, 5,6 milyon.\nİzmir, 4,4 milyon.\n\nSon."
        );
    }

    #[test]
    fn strip_think_block_code_fence_html_and_emoji() {
        let raw = "<think>\nKullanıcı ne istiyor? Düşüneyim.\n</think>\n\n\
               Merhaba 🚀 dünya 👩‍💻 ve 🇹🇷!\n\n\
               ```python\nprint('kod')\n```\n\n\
               <p>Bir <b>paragraf</b></p>1️⃣ Son.";
        assert_eq!(strip_format(raw), "Merhaba dünya ve!\n\nBir paragraf\nSon.");
        // Orphan closing tag: everything before it is reasoning.
        assert_eq!(strip_format("düşünce\n</think>\nCevap."), "Cevap.");
        // Case-insensitive, unclosed opener drops the rest.
        assert_eq!(strip_format("Cevap.\n<THINK>yarım kalmış"), "Cevap.");
    }

    #[test]
    fn strip_and_normalize_leave_plain_text_unchanged() {
        let plain = "Dün akşam İstanbul'dan Ankara'ya 450 km yol yaptık.\n\
                 Yolda 3,5 saat kaldık; hava 12 derece idi.\n\n\
                 Ertesi gün \"çok yorgunum\" dedi ve ışıkları kapattı. Ilık bir sabahtı (gerçekten).";
        assert_eq!(strip_format(plain), plain);
        assert_eq!(normalize(&strip_format(plain)), plain);
        // Only whitespace differs.
        let messy = "  Bir   cümle.\t\nİkinci   cümle.  ";
        assert_eq!(normalize(&strip_format(messy)), normalize(messy));
    }

    #[test]
    fn strip_keeps_mask_placeholders() {
        let masked = "<ÖZEL>'da <YIL> yılında <SAYI> kişi <URL> ve <EPOSTA> kullandı.";
        assert_eq!(strip_format(masked), masked);
    }

    // ---------------------------------------------------------------- openers

    #[test]
    fn each_opener_is_dropped() {
        let cases = [
            (
                "Elbette! Türkiye'nin başkenti Ankara'dır.",
                "Türkiye'nin başkenti Ankara'dır.",
            ),
            (
                "Tabii ki, işte cevabınız:\nAnkara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Tabi ki yardımcı olurum. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            ("Tabii. Ankara başkenttir.", "Ankara başkenttir."),
            ("Harika bir soru! Ankara başkenttir.", "Ankara başkenttir."),
            ("Harika soru. Ankara başkenttir.", "Ankara başkenttir."),
            ("Güzel bir soru. Ankara başkenttir.", "Ankara başkenttir."),
            ("İşte bazı öneriler:\nSu için.", "Su için."),
            (
                "Memnuniyetle açıklarım. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            ("Kesinlikle! Ankara başkenttir.", "Ankara başkenttir."),
            (
                "Sorunuz için teşekkürler. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Sorunuz için teşekkür ederim! Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Merhaba! Size nasıl yardımcı olabilirim? Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Merhaba, bu konuda yardımcı olabilirim. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            ("ELBETTE! Ankara başkenttir.", "Ankara başkenttir."),
            // Stacked openers are all removed.
            (
                "Elbette! İşte açıklama:\n\nAnkara başkenttir.",
                "Ankara başkenttir.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(drop_chatbot_opener(input), expected, "input: {input:?}");
            assert!(match_chatbot_opener(input).is_some(), "input: {input:?}");
        }
        // Every opener in the list is exercised above.
        assert_eq!(CHATBOT_OPENERS.len(), 13);
    }

    #[test]
    fn non_openers_are_kept() {
        for s in [
            "İşte bu yüzden geldim.",
            "Kesinlikle katılıyorum.",
            "Merhaba, ben Ali. Bugün hava güzel.",
            "Tabiiyet kanunu değişti.",
            "Ankara başkenttir.",
            "",
        ] {
            assert_eq!(drop_chatbot_opener(s), s, "input: {s:?}");
            assert!(match_chatbot_opener(s).is_none(), "input: {s:?}");
        }
    }

    #[test]
    fn opener_found_through_markdown_and_think() {
        let raw = "<think>plan</think>\n\n**Elbette!** İşte *liste*:\n\n- Bir";
        assert_eq!(preprocess(raw, &all()), "Bir");
    }

    // ---------------------------------------------------------------- mask

    #[test]
    fn mask_year_number_decimal() {
        assert_eq!(
            mask("bu 1999 ve 2023'te, 2150 ile 1899 değil; 3,14 ile 1.000.000 ve 12 oldu."),
            "bu <YIL> ve <YIL>'te, <SAYI> ile <SAYI> değil; <SAYI> ile <SAYI> ve <SAYI> oldu."
        );
        assert_eq!(mask("tarih 19.05.2023 idi"), "tarih <SAYI> idi");
        assert_eq!(mask("sene 2023."), "sene <YIL>.");
    }

    #[test]
    fn mask_urls_and_emails() {
        assert_eq!(
        mask("bkz. https://ornek.com/2023/yazi?id=5, ya da www.site.org'a bak; ali1990@posta.com.tr yaz."),
        "bkz. <URL>, ya da <URL>'a bak; <EPOSTA> yaz."
    );
    }

    #[test]
    fn mask_apostrophe_proper_noun_keeps_suffix() {
        assert_eq!(mask("İstanbul'da yaşıyorum."), "<ÖZEL>'da yaşıyorum.");
        assert_eq!(
            mask("dün Türkiye’nin en büyük şehri İzmir'e gittik."),
            "dün <ÖZEL>’nin en büyük şehri <ÖZEL>'e gittik."
        );
    }

    #[test]
    fn mask_sentence_initial_capital_is_not_masked() {
        assert_eq!(
            mask("Ankara güzel. Bugün Ali geldi!\nIşık yandı: Sonra Zeynep gitti."),
            "Ankara güzel. Bugün <ÖZEL> geldi!\nIşık yandı: Sonra <ÖZEL> gitti."
        );
    }

    #[test]
    fn mask_is_idempotent() {
        let s = "Ali 1990'da İstanbul'a geldi ve NATO raporunu https://x.com/a adresinden okudu.";
        let once = mask(s);
        assert_eq!(mask(&once), once);
    }

    // ---------------------------------------------------------------- pipeline

    const SAMPLES: &[&str] = &[
    "Elbette! İşte **Türkiye** hakkında bilgiler:\n\n## Genel\n\n- Başkent: *Ankara*\n- Nüfus: 85.000.000\n\n| Şehir | Yıl |\n|---|---|\n| İstanbul | 1453 |\n\nDaha fazlası için https://tr.wikipedia.org adresine bakın 🚀.",
    "<think>Kullanıcı Türkçe soruyor.</think>Merhaba! Size nasıl yardımcı olabilirim?\n\n1. Birinci — madde\n2. İkinci – madde…",
    "Dün akşam Kadıköy'de “harika” bir konser vardı; 2.500 kişi geldi. Ilık bir geceydi.",
    "— Nereye gidiyorsun? dedi Ahmet.\n— Eve, diye cevap verdi.",
    "＊＊Tam genişlik＊＊ metin ve ＃ başlık\n\n— madde bir\n— madde iki",
    "**1.** Madde ve `İstanbul`da` kod\n\n> > İç içe alıntı",
    "Fiyat 3,5 TL'den 4.250,75 TL'ye çıktı (%21). ali.veli@ornek.com.tr adresine yazın.",
    "",
    "   \n\n  ",
];

    #[test]
    fn preprocess_full_pipeline() {
        let raw = SAMPLES[0];
        assert_eq!(
        preprocess(raw, &all()),
        "Genel\n\nBaşkent: Ankara\nNüfus: <SAYI>\n\nŞehir, <ÖZEL>.\nİstanbul, <SAYI>.\n\nDaha fazlası için <URL> adresine bakın."
    );
    }

    #[test]
    fn preprocess_options_toggle_stages() {
        let raw = "Elbette! **Ankara** 1923'te başkent oldu.";
        let none = PreprocessOptions {
            strip: false,
            drop_opener: false,
            mask: false,
        };
        assert_eq!(preprocess(raw, &none), raw);
        let strip_only = PreprocessOptions {
            strip: true,
            ..none
        };
        assert_eq!(
            preprocess(raw, &strip_only),
            "Elbette! Ankara 1923'te başkent oldu."
        );
        assert_eq!(preprocess(raw, &all()), "Ankara <YIL>'te başkent oldu.");
    }

    #[test]
    fn preprocess_is_idempotent() {
        for raw in SAMPLES {
            let once = preprocess(raw, &all());
            let twice = preprocess(&once, &all());
            assert_eq!(twice, once, "raw: {raw:?}");
            let n = normalize(raw);
            assert_eq!(normalize(&n), n, "normalize not idempotent for {raw:?}");
        }
    }

    #[test]
    fn preprocess_batch_matches_single() {
        let docs: Vec<String> = (0..200)
            .map(|i| format!("{} #{i}", SAMPLES[i % SAMPLES.len()]))
            .collect();
        let batch = preprocess_batch(&docs, &all());
        assert_eq!(batch.len(), docs.len());
        for (d, b) in docs.iter().zip(&batch) {
            assert_eq!(&preprocess(d, &all()), b);
        }
    }

    #[test]
    fn no_panic_on_tricky_unicode() {
        let long_word = "x".repeat(10_000);
        let unmatched_stars = "*a ".repeat(3_000);
        let tricky = [
            "",
            "\u{0301}",
            "\u{0301}\u{0301}x",
            "👨‍👩‍👧‍👦👍🏽🏳️‍🌈🇹🇷1️⃣#️⃣",
            "مرحبا بالعالم 123 ٣٤٥",
            "שלום עולם",
            "\u{202E}tersine\u{202C}",
            "<<<>>>**__``~~||",
            "*",
            "|",
            "- ",
            "#",
            "```",
            "<think>",
            "</think>",
            "İ\u{0307}i\u{0307}",
            "a\u{0000}b\u{0007}c",
            "\u{FEFF}\u{200B}\u{200D}",
            "'''’’’",
            "http://",
            "@@@.com",
            "1.2.3.4.5.6.7.8.9",
            long_word.as_str(),
            unmatched_stars.as_str(),
            "漢字かなカナ한국어",
            "\u{10FFFF}\u{E0020}\u{E007F}",
        ];
        for s in tricky {
            let p = preprocess(s, &all());
            let _ = strip_format(s);
            let _ = mask(s);
            let _ = drop_chatbot_opener(s);
            assert_eq!(preprocess(&p, &all()), p, "not idempotent for {s:?}");
        }
    }

    #[test]
    fn strip_removes_latex_math_and_commands() {
        let cases = [
            "Sonuç \\(x^2 + 1\\) olur.",
            "Formül şudur: $$\\frac{a}{b} \\cdot c$$ ve devamı.",
            "Hız $v = \\frac{d}{t}$ formülüyle bulunur.",
            "Toplam \\[ \\sum_{i=1}^{n} i \\] eder.",
            "Burada \\text{Hız} = 5 alınır.",
        ];
        for c in cases {
            let out = strip_format(c);
            assert!(!out.contains('\\'), "backslash left in {out:?}");
            assert!(
                !out.contains("frac") && !out.contains("cdot") && !out.contains("sum"),
                "{out:?}"
            );
            assert!(!out.contains('{') && !out.contains('}'), "{out:?}");
        }
        assert!(strip_format("Burada \\text{Hız} = 5 alınır.").contains("Hız"));
    }

    #[test]
    fn strip_keeps_dollar_prices_and_plain_text() {
        let plain = "Fiyat $5 ve $10 arasında değişiyor, {not} kalsın.";
        assert_eq!(strip_format(plain), plain);
    }
}

/// Content features.
mod features {
    use akana_core::aidetect::features::{
        content_features, extract_features, extract_features_batch, FEATURE_NAMES, N_FEATURES,
    };
    use akana_core::aidetect::features::{
        FwMatch, AI_LEXICON, FUNCTION_WORD_FEATURES, HUMAN_LEXICON, N_FUNCTION_WORD_FEATURES,
    };
    use akana_core::aidetect::preprocess::{preprocess, PreprocessOptions};
    use akana_core::style::{ClicheCategory, ClicheMatcher};

    const EPS: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn feature(text: &str, name: &str) -> f64 {
        let idx = FEATURE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("unknown feature {name}"));
        extract_features(text)[idx]
    }

    #[test]
    fn feature_count_matches_names() {
        assert_eq!(N_FEATURES, 71);
        assert_eq!(FEATURE_NAMES.len(), N_FEATURES);
        assert_eq!(extract_features("").len(), FEATURE_NAMES.len());
        // New features are appended, so the existing order stays stable.
        assert_eq!(FEATURE_NAMES[24], "log_words");
        assert_eq!(FEATURE_NAMES[25], "aorist_ratio");
        assert_eq!(FEATURE_NAMES[66], "sents_per_para_mean");
        assert_eq!(FEATURE_NAMES[70], "atesman_readability");
        // The function-word block sits right after aorist_ratio, in FUNCTION_WORD_FEATURES order.
        assert_eq!(N_FUNCTION_WORD_FEATURES, 32);
        for (i, (name, _)) in FUNCTION_WORD_FEATURES.iter().enumerate() {
            assert_eq!(FEATURE_NAMES[26 + i], *name);
            assert!(name.is_ascii(), "{name}");
        }
        assert_eq!(extract_features("Bir cümle.").len(), FEATURE_NAMES.len());
        let named = content_features("Bir cümle.").named();
        assert_eq!(named.len(), FEATURE_NAMES.len());
        for ((n, _), expected) in named.iter().zip(FEATURE_NAMES) {
            assert_eq!(n, expected);
        }
        // Names are unique.
        let mut sorted = FEATURE_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), FEATURE_NAMES.len());
    }

    #[test]
    fn degenerate_inputs_are_finite_and_zero() {
        let inputs = [
            "",
            "   \n\n  ",
            "kelime",
            "🚀🔥😀",
            "<ÖZEL> <SAYI> <YIL>'da <URL> <EPOSTA>.",
            "...!?;:,",
            "\n",
        ];
        for text in inputs {
            let v = extract_features(text);
            assert_eq!(v.len(), N_FEATURES, "{text:?}");
            assert!(v.iter().all(|x| x.is_finite()), "{text:?}: {v:?}");
        }
        for text in ["", "🚀🔥😀", "<ÖZEL> <SAYI> <YIL>'da <URL> <EPOSTA>."] {
            let f = content_features(text);
            assert_eq!(f.log_words, 0.0, "{text:?}");
            assert_eq!(f.mattr_50, 0.0);
            assert_eq!(f.mean_word_len, 0.0);
            assert_eq!(f.oov_rate, 0.0);
        }
        let one = content_features("kelime");
        assert!(close(one.mattr_50, 1.0));
        assert!(close(one.hapax_ratio, 1.0));
        assert!(close(one.yules_k, 0.0));
        assert!(close(one.sent_len_mean, 1.0));
        assert_eq!(one.sent_len_cv, 0.0);
        assert!(close(one.log_words, 2f64.ln()));
    }

    #[test]
    fn comma_rate_and_sentence_cv() {
        // 2 sentences of 4 and 2 words; 2 commas.
        let f = content_features("elma, armut, kiraz geldi. hava güzel.");
        assert!(close(f.commas_per_sentence, 1.0), "{f:?}");
        assert!(close(f.sent_len_mean, 3.0));
        // population std of [4, 2] = 1, mean 3.
        assert!(close(f.sent_len_cv, 1.0 / 3.0));
        assert_eq!(f.long_sent_share, 0.0);
        assert!(close(f.log_words, 7f64.ln()));
    }

    #[test]
    fn punctuation_rates() {
        let f = content_features("Geldin mi? Evet geldim! Şunu bil: hayat kısa; dikkat et.");
        // Sentences: "Geldin mi?", "Evet geldim!", "Şunu bil: hayat kısa; dikkat et."
        assert!(close(f.questions_per_100s, 100.0 / 3.0), "{f:?}");
        assert!(close(f.exclamations_per_100s, 100.0 / 3.0));
        // 10 words.
        assert!(close(f.semicolons_per_100w, 10.0));
        assert!(close(f.colons_per_100w, 10.0));
        // A colon inside a masked time (no following space) is not counted.
        let g = content_features("Saat <SAYI>:<SAYI> oldu.");
        assert_eq!(g.colons_per_100w, 0.0);
    }

    #[test]
    fn paragraph_cv_and_lines() {
        // Paragraph lengths 2 and 6 -> mean 4, std 2, CV 0.5. A heading line without final
        // punctuation is its own sentence.
        let f = content_features("Başlık burada\n\nbir iki üç. dört beş altı.");
        assert!(close(f.para_len_cv, 0.5), "{f:?}");
        assert!(close(f.sent_len_mean, 8.0 / 3.0));
        assert_eq!(content_features("tek paragraf burada.").para_len_cv, 0.0);
    }

    #[test]
    fn vocabulary_statistics() {
        // a a b c: types 3, hapax 2 (b, c), sum f^2 = 4 + 1 + 1 = 6, N = 4.
        let f = content_features("kedi kedi köpek kuş");
        assert!(close(f.hapax_ratio, 2.0 / 3.0));
        assert!(close(f.yules_k, 1e4 * (6.0 - 4.0) / 16.0));
        assert!(close(f.mattr_50, 3.0 / 4.0));
        assert!(close(f.mean_word_len, (4.0 + 4.0 + 5.0 + 3.0) / 4.0));
        // "ve" and "bu" are function words.
        let g = content_features("kedi ve bu köpek");
        assert!(close(g.lexical_density, 0.5));
        assert!(close(g.ve_per_100w, 25.0));
    }

    #[test]
    fn mattr_uses_window_of_50() {
        // 100 distinct words: every window is all-distinct.
        let words: Vec<String> = (0..100)
            .map(|i| format!("kelime{}", "a".repeat(i + 1)))
            .collect();
        let f = content_features(&words.join(" "));
        assert!(close(f.mattr_50, 1.0));
        // Two alternating words: every window of 50 holds 2 types.
        let g = content_features(&"elma armut ".repeat(60));
        assert!(close(g.mattr_50, 2.0 / 50.0));
    }

    #[test]
    fn mask_tokens_are_excluded_from_lexical_statistics() {
        let plain = content_features("kitap okundu ve ayrıca beğenildi.");
        let masked = content_features(
            "<ÖZEL> kitap <SAYI> okundu <YIL>'da ve <URL> ayrıca <EPOSTA> beğenildi <ÖZEL>'nin.",
        );
        assert!(close(plain.mattr_50, masked.mattr_50));
        assert!(close(plain.hapax_ratio, masked.hapax_ratio));
        assert!(close(plain.yules_k, masked.yules_k));
        assert!(close(plain.mean_word_len, masked.mean_word_len));
        assert!(close(plain.lexical_density, masked.lexical_density));
        assert!(close(plain.ai_lex_per_100w, masked.ai_lex_per_100w));
        assert!(close(plain.ve_per_100w, masked.ve_per_100w));
        assert!(close(plain.oov_rate, masked.oov_rate));
        assert!(close(plain.log_words, masked.log_words));
        // ...but they still count as structural tokens for sentence length.
        assert!(close(plain.sent_len_mean, 5.0));
        assert!(close(masked.sent_len_mean, 11.0));
        // The apostrophe suffix of a mask is not a word either.
        assert_eq!(content_features("<ÖZEL>'da").log_words, 0.0);
    }

    #[test]
    fn turkish_casing_is_respected() {
        // I -> ı and İ -> i before matching.
        let f = content_features("AYRICA DİYE IŞIK");
        assert!(close(f.ai_lex_per_100w, 100.0 / 3.0), "{f:?}");
        assert!(close(f.human_lex_per_100w, 100.0 / 3.0));
        // "IŞIK" and "ışık" are one type; "ISIK" would not be.
        let g = content_features("IŞIK ışık");
        assert!(close(g.hapax_ratio, 0.0));
        assert!(close(g.mattr_50, 0.5));
        let h = content_features("İNSAN insan");
        assert!(close(h.mattr_50, 0.5));
        // Decomposed İ (i + U+0307) from a non-Turkish lower-casing is folded too.
        let d = content_features("i\u{307}nsan insan");
        assert!(close(d.mattr_50, 0.5));
    }

    #[test]
    fn word_list_prefix_matching() {
        // artır matches inflected and derived forms; short exact entries do not overmatch.
        let f = content_features("artırılması artırmak güçlendirildi vurguladı");
        assert!(close(f.ai_lex_per_100w, 100.0), "{f:?}");
        let g = content_features("şubat şunu hiçbir hiç yinelemek yine");
        // hiçbir (prefix entry), hiç, yine -> 3 of 6; "şubat", "yinelemek" do not match "şu"/"yine".
        assert!(close(g.human_lex_per_100w, 50.0), "{g:?}");
    }

    #[test]
    fn question_clitic_matching() {
        let f = content_features("geldin mi? gelecek misin? müdür geldi mü?");
        // mi, misin, mü -> 3 of 7 words; "müdür" is not the clitic.
        assert!(close(f.human_lex_per_100w, 300.0 / 7.0), "{f:?}");
    }

    #[test]
    fn suffix_heuristics() {
        let f = content_features("gelmiş yapılmaktadır olduğunu söylendi kullanan yetmiş");
        // gelmiş (yetmiş is a numeral)
        assert!(close(f.mis_per_100w, 100.0 / 6.0), "{f:?}");
        // yapılmaktadır
        assert!(close(f.makta_per_100w, 100.0 / 6.0));
        // olduğunu
        assert!(close(f.dik_per_100w, 100.0 / 6.0));
        // yapılmaktadır, söylendi (kullanan is active)
        assert!(close(f.passive_per_100w, 200.0 / 6.0));
        // 1st person plural past is not a -DIk participle.
        assert_eq!(content_features("geldik gittik").dik_per_100w, 0.0);
        assert!(content_features("geldikten sonra").dik_per_100w > 0.0);
    }

    #[test]
    fn cliches_are_counted() {
        let f = content_features("Bu çalışma kritik bir rol oynamaktadır.");
        assert!(f.cliche_per_100w > 0.0, "{f:?}");
        assert_eq!(content_features("Bugün hava güzel.").cliche_per_100w, 0.0);
    }

    #[test]
    fn oov_proxy() {
        let known = content_features("kitaplarımızdan geliyorum evlerde ağacı");
        assert!(known.oov_rate < EPS, "{known:?}");
        let f = content_features("qwxzt kitap gelmis yapmıs");
        // qwxzt and the missing-diacritic forms gelmis / yapmıs are OOV.
        assert!(close(f.oov_rate, 0.75), "{f:?}");
    }

    #[test]
    fn deterministic_and_batch_equals_single() {
        let docs: Vec<String> = [
            "Ayrıca, bu süreç kritik bir rol oynamaktadır. Özellikle verimlilik artırılmalıdır.",
            "Ben de dün gittim ama hiç beğenmedim yani. Şu işe bak!",
            "",
            "<ÖZEL> <YIL>'da geldi.",
            "Başlık\n\nParagraf bir. Paragraf iki uzun, çok uzun bir cümle.",
        ]
        .repeat(20)
        .iter()
        .map(|s| s.to_string())
        .collect();
        let single: Vec<Vec<f64>> = docs.iter().map(|d| extract_features(d)).collect();
        let again: Vec<Vec<f64>> = docs.iter().map(|d| extract_features(d)).collect();
        assert_eq!(single, again);
        assert_eq!(extract_features_batch(&docs), single);
        assert!(extract_features_batch::<String>(&[]).is_empty());
    }

    #[test]
    fn works_on_preprocessed_markdown() {
        let raw = "Elbette! İşte **önemli** noktalar:\n\n- Ayrıca, 2023 yılında artış oldu.\n- İstanbul'da 5 kişi geldi.";
        let clean = preprocess(raw, &PreprocessOptions::default());
        let v = extract_features(&clean);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(feature(&clean, "ai_lex_per_100w") > 0.0);
    }

    #[test]
    fn cliche_gate_matches_cliche_matcher() {
        // One sentence per pattern of the three categories, plus near misses.
        let texts = [
            "Bu çalışma kritik bir rol oynamaktadır.",
            "Eğitim hayati bir önem taşımaktadır.",
            "Bu konu büyük bir öneme sahiptir.",
            "Teknoloji vazgeçilmez bir unsur oldu.",
            "Bu sadece bir araç değil, aynı zamanda bir yaşam biçimidir.",
            "Yalnızca ekonomi değil, bununla birlikte kültür de etkilendi.",
            "Etkisi turizm ile sınırlı kalmayıp tarımı da kapsadı.",
            "Proje yerel olmakla birlikte, ulusal bir niteliğini de taşımaktadır.",
            "Ekonominin ötesinde, konunun sosyal boyutu da var.",
            "Bu durum göz ardı edilmemelidir.",
            "Unutulmamalıdır ki her şey değişir.",
            "Özetlemek gerekirse sonuç iyi.",
            "Bu önemli bir adım oldu ve temel bir dönüm noktası sayıldı.",
            "KRITIK BIR ROL OYNAMAKTADIR.",
            "Bu değil, aynı zamanda o.",
            "Rol büyük. Bir önem yok. Bir çalışma yaptık.",
            "Bugün hava güzel.",
        ];
        for text in texts {
            let words = text
                .split(|c: char| !c.is_alphabetic())
                .filter(|w| !w.is_empty())
                .count() as f64;
            let direct = ClicheMatcher::find_all(text)
                .iter()
                .filter(|m| {
                    matches!(
                        m.category,
                        ClicheCategory::ConclusionHedge
                            | ClicheCategory::RhetoricalCalque
                            | ClicheCategory::IndefiniteArticleInflation
                    )
                })
                .count() as f64;
            let got = content_features(text).cliche_per_100w;
            assert!(
                close(got, 100.0 * direct / words),
                "{text:?}: {got} vs {direct}"
            );
        }
        assert!(content_features(texts[0]).cliche_per_100w > 0.0);
    }

    // ------------------------------------------------------------------- D-005 changes

    /// Words of a cleaned text, as `log_words` counts them.
    fn n_words(clean: &str) -> f64 {
        content_features(clean).log_words.exp_m1().round()
    }

    #[test]
    fn label_colons_are_not_content_colons() {
        // A line-leading label is list residue, not a content colon.
        let text = "Adım 1: Malzemeleri hazırlayın.";
        assert_eq!(content_features(text).colons_per_100w, 0.0);

        // A mid-sentence colon still counts: 1 colon over 3 words.
        let mid = "Şunu söyledi: gel";
        assert!(close(content_features(mid).colons_per_100w, 100.0 / 3.0));
        // A finite verb before the colon makes it a clause, not a label.
        assert!(close(
            content_features("Ali dedi: gel").colons_per_100w,
            100.0 / 3.0
        ));
        assert!(
            content_features("Dosya incelendi, gereği düşünüldü: onandı.").colons_per_100w > 0.0
        );
        // So does a lead-in longer than four words.
        assert!(
            content_features("Bu konuda dikkat edilmesi gereken noktalar: zaman ve para.")
                .colons_per_100w
                > 0.0
        );
        // A label after a finished sentence counts as a label too.
        assert_eq!(
            content_features("Her şey bitti. Sonuç: başarılı bir deneme.").colons_per_100w,
            0.0
        );
        // Only the first colon of a sentence can end a label: 1 of 2 colons over 4 words.
        assert!(close(
            content_features("Ad: Ali, soyad: Veli.").colons_per_100w,
            25.0
        ));
        // Possessive nouns ending in -DI are not past-tense verbs.
        assert_eq!(
            content_features("Ürün fiyatı: yüz lira.").colons_per_100w,
            0.0
        );
    }

    #[test]
    fn label_colons_are_left_out_of_content() {
        let raw = "Elbette! İşte tarif:\n\n- **Adım 1:** Malzemeleri hazırlayın.\n\
               - **Adım 2:** Fırını 180 dereceye ısıtın.\n1. Hazırlık: Kabı yağlayın.\n\n\
               Bir not daha: şunu unutmayın: sabır gerekir.\n\n**Sonuç:** Afiyet olsun.";
        let clean = preprocess(raw, &PreprocessOptions::default());
        let total = clean
            .match_indices(':')
            .filter(|&(i, _)| {
                clean[i + 1..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
            })
            .count() as f64;
        let content = content_features(&clean).colons_per_100w * n_words(&clean) / 100.0;
        // 6 colons; the labels Adım 1, Adım 2, Hazırlık, Bir not daha, Sonuç are left out.
        assert!(close(total, 6.0), "{clean:?}");
        assert!(close(content, 1.0), "{clean:?}");
    }

    #[test]
    fn pruned_juzek_words_are_not_matched() {
        // D-005: reversed (ters), unsupported (tutmuyor) and too-rare (yetersiz) Juzek lemmas.
        const PRUNED: [&str; 13] = [
            "detaylı",
            "takma",
            "dolayısıyla",
            "ilişkin",
            "itibaren",
            "zedele",
            "titizlik",
            "aksaklık",
            "uyul",
            "çaplı",
            "dir",
            "sürdürülebilik",
            "sıkılaş",
        ];
        for w in PRUNED {
            assert!(
                !AI_LEXICON.iter().chain(HUMAN_LEXICON).any(|e| e.word == w),
                "{w} is still listed"
            );
        }
        assert_eq!(AI_LEXICON.len(), 31);
        assert_eq!(HUMAN_LEXICON.len(), 36);
        let f = content_features(
            "detaylı takma dolayısıyla ilişkin itibaren zedeledi titizlikle aksaklıklar \
         uyulmalı çaplı dir sürdürülebilik sıkılaştı",
        );
        assert_eq!(f.ai_lex_per_100w, 0.0, "{f:?}");
        assert_eq!(f.human_lex_per_100w, 0.0, "{f:?}");
        // Kept entries still match; `dolayı` is exact now, so `dolayısıyla` above did not.
        let g = content_features("ayrıca dolayı vurguladı");
        assert!(close(g.ai_lex_per_100w, 200.0 / 3.0), "{g:?}");
        assert!(close(g.human_lex_per_100w, 100.0 / 3.0), "{g:?}");
    }

    #[test]
    fn aorist_ratio_on_sentence_final_predicates() {
        // gelir, gider: aorist; bakıyor: progressive.
        assert!(close(
            content_features("Gelir. Gider. Bakıyor.").aorist_ratio,
            2.0 / 3.0
        ));
        // Person endings, the -mAz negative, a -lA verb and the potential are aorist.
        let f = content_features(
            "Her gün erken kalkarım. Bunu sevmez. Bu yöntem verimliliği sağlar. \
         Sonuç değişebilir. Çocuklar oynuyorlar.",
        );
        assert!(close(f.aorist_ratio, 0.8), "{f:?}");
        // The predicate before a question clitic counts.
        assert!(close(
            content_features("Gelir misin? Geliyor musun?").aorist_ratio,
            0.5
        ));
        // -tIr after a vowel is the aorist of a t-final stem, not the copula.
        assert!(close(
            content_features("Kuş uçar. Balık yüzer. Öğretmen anlatır.").aorist_ratio,
            1.0
        ));
        // Not aorist: copula -DIr, past, -mAktAdIr, plural nouns, listed non-verbs, masks.
        for text in [
            "Bu çok önemlidir.",
            "Bu ürüne sahiptir.",
            "Dün geldi.",
            "Süreç devam etmektedir.",
            "Kitaplar.",
            "Hepsi bu kadar.",
            "Çok teşekkürler.",
            "Tamam.",
            "Toplam <SAYI>.",
            "Yağmur yağıyor.",
            "",
        ] {
            assert_eq!(content_features(text).aorist_ratio, 0.0, "{text:?}");
        }
    }

    // ------------------------------------------------------------------- stylometry groups

    #[test]
    fn function_word_profile() {
        // 12 words: ben, de x2, için, çünkü, sen, mi; "bunu" and "senin" are not exact forms.
        let text = "Ben de bunu senin için yaptım çünkü sen de istedin mi bilmem.";
        for (name, count) in [
            ("fw_ben", 1.0),
            ("fw_de", 2.0),
            ("fw_icin", 1.0),
            ("fw_cunku", 1.0),
            ("fw_sen", 1.0),
            ("fw_mi", 1.0),
            ("fw_bu", 0.0),
            ("fw_kendi", 0.0),
        ] {
            assert!(close(feature(text, name), count * 100.0 / 12.0), "{name}");
        }
        // `da` is `fw_de`; inflected question clitics count, `müdür` does not; `kendini` is
        // not `kendi`.
        let g = "o da gelir mi misiniz müdür kendini kendi";
        assert!(close(feature(g, "fw_o"), 100.0 / 8.0));
        assert!(close(feature(g, "fw_de"), 100.0 / 8.0));
        assert!(close(feature(g, "fw_mi"), 200.0 / 8.0));
        assert!(close(feature(g, "fw_kendi"), 100.0 / 8.0));
        // Turkish casing: İÇİN -> için, DEĞİL -> değil.
        let h = "İÇİN ÇÜNKÜ DEĞİL";
        for name in ["fw_icin", "fw_cunku", "fw_degil"] {
            assert!(close(feature(h, name), 100.0 / 3.0), "{name}");
        }
        // ve / ile are their own features, not part of the profile.
        for (name, m) in FUNCTION_WORD_FEATURES {
            if let FwMatch::Forms(forms) = m {
                assert!(!forms.contains(&"ve") && !forms.contains(&"ile"), "{name}");
            }
        }
        let f = content_features("");
        assert!(f.function_words_per_100w.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sentence_start_case_and_terminal_punctuation() {
        // Sentences: "bugün geldim." (lower), "Yarın giderim" (no end mark), "nasılsın?" (lower).
        let f = content_features("bugün geldim. Yarın giderim\nnasılsın?");
        assert!(close(f.lower_start_share, 2.0 / 3.0), "{f:?}");
        assert!(close(f.no_terminal_punct_share, 1.0 / 3.0), "{f:?}");
        // Closing quotes after the end mark are skipped; a mask start is not lower-case.
        let g = content_features("<ÖZEL> geldi. Ali \"tamam.\" Dedi ki \"evet!\"");
        assert_eq!(g.lower_start_share, 0.0, "{g:?}");
        assert_eq!(g.no_terminal_punct_share, 0.0, "{g:?}");
        assert_eq!(content_features("").lower_start_share, 0.0);
        assert_eq!(content_features("").no_terminal_punct_share, 0.0);
    }

    #[test]
    fn word_length_shares() {
        // ev, bir, gel: <= 3 letters; kitaplarım: 10 letters; evde (4) and kelimeler (9): neither.
        let f = content_features("ev bir gel evde kelimeler kitaplarım <ÖZEL>");
        assert!(close(f.short_word_share, 3.0 / 6.0), "{f:?}");
        assert!(close(f.long_word_share, 1.0 / 6.0), "{f:?}");
        assert_eq!(content_features("<SAYI>").short_word_share, 0.0);
    }

    #[test]
    fn punctuation_combinations_per_100_sentences() {
        // 5 sentences, one run of each kind.
        let f = content_features("Gel!! Ne?! Hmm... Nasıl?? Tamam.");
        assert!(close(f.multi_excl, 20.0), "{f:?}");
        assert!(close(f.interrobang, 20.0), "{f:?}");
        assert!(close(f.ellipsis, 20.0), "{f:?}");
        assert!(close(f.multi_q, 20.0), "{f:?}");
        // Mixed runs are interrobangs only; single marks and `..` count nowhere; `…` is an
        // ellipsis.
        let g = content_features("Ne!? Gel!!? Tamam! Nasıl? Bitti.. Peki…");
        assert!(
            g.interrobang > 0.0 && g.multi_excl == 0.0 && g.multi_q == 0.0,
            "{g:?}"
        );
        assert!(close(g.interrobang, 2.0 * g.ellipsis), "{g:?}");
        assert_eq!(content_features("Tamam! Nasıl? Bitti.").ellipsis, 0.0);
    }

    #[test]
    fn sentences_per_paragraph() {
        assert!(close(
            content_features("Geldi. Gitti.\n\nKaldı.").sents_per_para_mean,
            1.5
        ));
        assert!(close(
            content_features("Tek cümle burada.").sents_per_para_mean,
            1.0
        ));
        assert_eq!(content_features("").sents_per_para_mean, 0.0);
    }

    #[test]
    fn word_entropy_is_length_normalised() {
        // Two types, two tokens each: ln 2 / ln 4.
        assert!(close(content_features("ev göl ev göl").word_entropy, 0.5));
        // All tokens distinct: maximal.
        assert!(close(content_features("ev göl dağ yol").word_entropy, 1.0));
        assert_eq!(content_features("ev").word_entropy, 0.0);
        assert_eq!(content_features("").word_entropy, 0.0);
    }

    #[test]
    fn heaps_exponent_tracks_vocabulary_growth() {
        let distinct = "bir iki üç dört beş altı yedi sekiz dokuz on onbir oniki";
        assert!(close(content_features(distinct).heaps_exponent, 1.0));
        let repeated = "ev ev ev ev ev ev ev ev ev ev ev ev";
        assert!(close(content_features(repeated).heaps_exponent, 0.0));
        // Below 10 words the fit is not attempted.
        assert_eq!(content_features("ev göl dağ").heaps_exponent, 0.0);
    }

    #[test]
    fn compression_ratio_is_lower_for_repetitive_text() {
        let repetitive = content_features(&"Bu çok güzel bir gün. ".repeat(20)).compression_ratio;
        let varied = content_features(
            "Sabah erkenden kalktık, kahvaltıdan sonra sahile indik. Rüzgâr sertti ama deniz \
         sakindi. Öğleden sonra müzeyi gezdik; akşam da eski bir lokantada balık yedik.",
        )
        .compression_ratio;
        assert!(
            repetitive > 0.0 && repetitive < varied,
            "{repetitive} vs {varied}"
        );
        assert_eq!(content_features("").compression_ratio, 0.0);
    }

    #[test]
    fn atesman_readability_formula() {
        // "ev" 1 syllable, "güzel" 2: 198.825 - 40.175 * 1.5 - 2.610 * 2.
        assert!(close(
            content_features("Ev güzel.").atesman_readability,
            198.825 - 40.175 * 1.5 - 2.610 * 2.0
        ));
        assert_eq!(content_features("").atesman_readability, 0.0);
    }
}

/// Tree model, patterns, token transitions, report.
mod detection {
    use akana_core::aidetect::{
        ai_feature_vector, ai_probability, ai_text_detect, berturk_tokens, out_of_scope, AiVerdict,
        MODEL,
    };

    // Sample texts taken from the frozen golden sets (akana-data/golden), not written by hand.
    // OpenAI golden v1, wildchat_tr (o1-preview-2024-09-12), id 151b38df05e571c8
    const AI_TEXT: &str = r#"En katma değerli ve ölçeklenebilir kompozit malzemeler arasında **karbon fiber takviyeli polimerler (CFRP)** öne çıkmaktadır. CFRP, yüksek mukavemet/ağırlık oranı, mükemmel yorulma direnci ve korozyon direnci gibi üstün mekanik özellikleri sayesinde havacılık ve uzay sanayii, otomotiv, rüzgar enerjisi ve spor ekipmanları gibi çeşitli endüstrilerde yaygın olarak kullanılmaktadır.

Bu malzemelerin üretimi, reçine aktarım kalıplama (RTM) ve otoklav laminasyonu gibi ölçeklenebilir üretim teknikleriyle gerçekleştirilebilir. Ayrıca, otomasyon ve robotik teknolojilerinin entegrasyonu ile üretim süreçleri daha verimli hale getirilebilir, bu da büyük ölçekli üretime olanak tanır.

Karbon fiber kompozitlerin yüksek katma değeri, ürünlerin performansını artırırken ağırlığı azaltması ve enerji verimliliğine katkıda bulunmasından kaynaklanmaktadır. Bu özellikler, özellikle yakıt tasarrufu ve azaltılmış emisyonların önemli olduğu sektörlerde büyük ekonomik ve çevresel faydalar sağlamaktadır."#;

    // Claude golden v2, Turkish Wikisource poem (public domain), id 984244ffc4199c8c
    const POEM: &str = r#"İki keklik bir kayada ötüyor
Ötme de keklik derdim bana yetiyor,
Aman güzelim yetiyor.
Bu hasretlik bu ayrılık yetiyor
Yazma da müstantik uğraşırım kız benim
Aman güzelim kız benim
Mantosu oymalı iskarpini boyalı kız benim
Aman güzelim kız benim
İki de keklik bir kayada su içer
Dertli de kerem dertsizlere dert açar
Aman güzelim dert açar
Buna da asrı sevda derler tez geçer
Yazma da müstantik uğraşırım kız benim
Aman güzelim kız benim
Mantosu oymalı iskarpini boyalı kız benim
Aman güzelim kız benim"#;

    // OpenAI golden v1, wildchat_tr (gpt-4o-2024-08-06), id d3d3a89f0be18699: a chat answer the academic rule flags
    const ACADEMIC_CUES_TEXT: &str = r#"Elbette, Chain-of-Thought Prompting yöntemiyle adım adım inceleyelim:

1. **"Narrator" kelimesi**: 
   - "Narrator" kelimesindeki harflere bakalım: N, a, r, r, a, t, o, r.
   - Burada üç tane 'r' harfi var.

2. **"Strawberry" kelimesi**:
   - "Strawberry" kelimesindeki harflere bakalım: S, t, r, a, w, b, e, r, r, y.
   - Burada da üç tane 'r' harfi var.

3. **"Kelimeler arasındaki toplam 'r' sayısı"**: 
   - "Narrator" kelimesinde 3 'r' + "Strawberry" kelimesinde 3 'r' = Toplam 6 'r' harfi bulunuyor.

Sonuç olarak, "Narrator" ve "Strawberry" kelimelerindeki toplam 'r' harfi sayısı 6."#;

    #[test]
    fn model_loads() {
        assert_eq!(MODEL.features.len(), 75);
        assert!(MODEL.threshold > 0.5 && MODEL.threshold < 1.0);
        assert!(!MODEL.patterns.is_empty());
    }

    #[test]
    fn berturk_pieces() {
        // Same pieces as Hugging Face BertWordPieceTokenizer on the flattened text.
        let t = berturk_tokens(
            "**Merhaba** Dünya, 2024'te İstanbul'a gittim. Kullanılabilirlik değerlendirmesi",
        );
        let want = [
            "merhaba",
            "dünya",
            ",",
            "0",
            "'",
            "te",
            "istanbul",
            "'",
            "a",
            "gittim",
            ".",
            "kullanılabilir",
            "##lik",
            "değerlendirmesi",
        ];
        assert_eq!(t, want);
    }

    #[test]
    fn contributions_add_up() {
        let a = ai_text_detect(AI_TEXT);
        let sum: f64 = a.features.iter().map(|f| f.contribution).sum();
        assert!((a.bias + sum - a.raw_score).abs() < 1e-6);
        assert!(a.ai_probability > 0.0 && a.ai_probability < 1.0);
        assert_eq!(a.features.len(), 75);
    }

    #[test]
    fn short_text_is_not_judged() {
        let a = ai_text_detect("Elbette, yardımcı olabilirim.");
        assert_eq!(a.verdict, AiVerdict::TooShort);
        assert!(a.report.clone().contains("kelime"));
    }

    #[test]
    fn ai_like_text_report() {
        let a = ai_text_detect(AI_TEXT);
        assert_eq!(a.verdict, AiVerdict::Ai);
        assert!(a.patterns.len() >= 3);
        let r = a.report.clone();
        assert!(r.contains("AI tespit raporu"));
    }

    #[test]
    fn poem_is_out_of_scope() {
        assert_eq!(out_of_scope(POEM).map(|s| s.kind), Some("şiir".to_string()));
        assert_eq!(ai_text_detect(POEM).verdict, AiVerdict::OutOfScope);
        assert!(out_of_scope(AI_TEXT).is_none());
        // academic text is only flagged, the decision is still made
        let a = ai_text_detect(ACADEMIC_CUES_TEXT);
        assert_eq!(a.scope.map(|s| s.kind), Some("akademik".to_string()));
        assert_ne!(a.verdict, AiVerdict::OutOfScope);
    }

    /// Rust vs Python reference on 400 golden-set texts; the fixture is generated outside the
    /// repository (`tr-ai-detect/scripts/export_akana_model.py`), set `AKANA_AIDETECT_FIXTURE`
    /// to its path. Skipped when absent.
    #[test]
    fn parity_with_python_reference() {
        let Ok(path) = std::env::var("AKANA_AIDETECT_FIXTURE") else {
            return;
        };
        let rows: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let names: Vec<&str> = MODEL.features.iter().map(|f| f.key.as_str()).collect();
        let (mut bad_p, mut worst) = (0, 0.0f64);
        for (i, r) in rows.iter().enumerate() {
            let text = r["text"].as_str().unwrap();
            let x = ai_feature_vector(text);
            for (j, want) in r["x"].as_array().unwrap().iter().enumerate() {
                let want = want.as_f64().unwrap();
                if (x[j] - want).abs() > 1e-6 * (1.0 + want.abs()) {
                    eprintln!(
                        "row {i} feature {}: rust {} python {}",
                        names[j], x[j], want
                    );
                }
            }
            let p = ai_probability(text);
            let d = (p - r["p"].as_f64().unwrap()).abs();
            worst = worst.max(d);
            if d > 1e-6 {
                bad_p += 1;
            }
        }
        eprintln!(
            "rows {} probability mismatches {bad_p} worst {worst:e}",
            rows.len()
        );
        assert_eq!(bad_p, 0);
    }
}
