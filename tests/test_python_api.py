"""
Comprehensive Python unit test suite for Akana.
"""

import akana


def test_casing_and_harmony():
    # Turkish specific casing
    assert akana.to_turkish_lower("İSTANBUL") == "istanbul"
    assert akana.to_turkish_lower("ILIK") == "ılık"
    assert akana.to_turkish_upper("istanbul") == "İSTANBUL"
    assert akana.to_turkish_upper("ılık") == "ILIK"
    assert akana.to_turkish_title("istanbul ve ankara") == "İstanbul Ve Ankara"

    # Vowel harmony
    assert akana.check_major_vowel_harmony("okul") is True
    assert akana.check_major_vowel_harmony("çiçek") is True
    assert akana.check_major_vowel_harmony("kitap") is False

    assert akana.check_minor_vowel_harmony("çocuk") is True
    assert akana.check_minor_vowel_harmony("odun") is True


def test_normalization():
    # Asciification
    assert akana.asciify("Türkçe, Çağdaş, Şiir, Ağaç") == "Turkce, Cagdas, Siir, Agac"

    # Deasciification
    assert (
        akana.deasciify("turkce nlp cok hizli calisiyor")
        == "türkçe nlp çok hızlı çalışıyor"
    )
    assert (
        akana.deasciify("ogrenci kutuphanede kitap okuyor")
        == "öğrenci kütüphanede kitap okuyor"
    )

    # Informal text normalization
    assert (
        akana.normalize_informal("yapcam dedim ve geliyom")
        == "yapacağım dedim ve geliyorum"
    )
    assert akana.normalize_informal("nooldu ya") == "ne oldu ya"
    assert akana.normalize_informal("çooookk güzel") == "çok güzel"


def test_morphology_analysis():
    morph = akana.Morphology()

    # Noun with voicing and possessive
    parses = morph.analyze("kitabım")
    assert len(parses) > 0
    top = parses[0]
    assert top["root"] == "kitap"
    assert "P1sg" in top["morphemes"]
    # Zemberek's format, with every morpheme and its surface, next to akana's tags.
    assert top["analysis"] == "[kitap:Noun] kitab:Noun+A3sg+ım:P1sg"
    assert top["pos"] == "Noun"

    # The lemma as the dictionary spells it, beside the folded key sparse terms use.
    name = morph.analyze("Ankara'dan")[0]
    assert (name["lemma"], name["display_lemma"]) == ("ankara", "Ankara")
    assert name["analysis"] == "[Ankara:Noun,Prop] ankara:Noun+A3sg+dan:Abl"
    paper = morph.analyze("kâğıdı")[0]
    assert (paper["lemma"], paper["display_lemma"]) == ("kağıt", "kâğıt")
    assert paper["nonstandard"] is False

    # Old orthography reads with the modern lemma, marked as no standard spelling.
    old = morph.analyze("karariyle")[0]
    assert (old["lemma"], old["nonstandard"]) == ("karar", True)

    # A participle reads as a noun ("that it is") and as an adjective ("which it is"): the
    # same akana tags, told apart by pos and the Zemberek analysis.
    olduğu = [p for p in morph.analyze("olduğu") if p["formatted"] == "[olmak:Verb] ol:Verb+PastPart+P3sg"]
    assert [p["pos"] for p in olduğu] == ["Noun", "Adj"]
    chosen = akana.Disambiguator().disambiguate(["Mahkemenin", "verdiği", "karar", "kesinleşti", "."])
    assert chosen[1]["pos"] == "Adj"

    # Noun with vowel drop
    parses_burnum = morph.analyze("burnum")
    assert len(parses_burnum) > 0
    assert any(p["root"] == "burun" for p in parses_burnum)

    # Noun with consonant doubling
    parses_hak = morph.analyze("hakkım")
    assert len(parses_hak) > 0
    assert any(p["root"] == "hak" for p in parses_hak)

    # Verb with progressive tense
    parses_gel = morph.analyze("geliyorum")
    assert len(parses_gel) > 0
    assert parses_gel[0]["root"] == "gel"
    assert "Prog" in parses_gel[0]["morphemes"]

    # Relative clitic -ki
    parses_evdeki = morph.analyze("evdeki")
    assert len(parses_evdeki) > 0
    assert any("RelClitic" in p["morphemes"] for p in parses_evdeki)

    # Diminutive -cik
    parses_evcik = morph.analyze("evcik")
    assert len(parses_evcik) > 0
    assert any("Dim" in p["morphemes"] for p in parses_evcik)


def test_compound_decomposer():
    analyses = akana.decompose_compound("denizaltı")
    assert len(analyses) > 0
    assert analyses[0]["part1"] == "deniz"
    assert analyses[0]["part2"] == "altı"

    analyses2 = akana.decompose_compound("başbakanlar")
    assert len(analyses2) > 0
    assert analyses2[0]["part1"] == "baş"
    assert analyses2[0]["part2"] == "bakanlar"

    # Two words that merely happen to make a third are no compound (mah + keme).
    assert akana.decompose_compound("mahkeme") == []


def test_dynamic_dictionary_loading():
    morph = akana.Morphology()
    morph.load_dictionary_str("blokzincir blokzincir Noun\nkuvars kuvars Noun")

    parses = morph.analyze("blokzincir")
    assert len(parses) > 0
    assert parses[0]["root"] == "blokzincir"


def test_morphology_generation():
    morph = akana.Morphology()

    # Kitabıma
    surface1 = morph.generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])
    assert surface1 == "kitabıma"

    # Burnum
    surface2 = morph.generate("burun", ["Noun", "A3sg", "P1sg"])
    assert surface2 == "burnum"

    # Hakkım
    surface3 = morph.generate("hak", ["Noun", "A3sg", "P1sg"])
    assert surface3 == "hakkım"

    # Geliyorum
    surface4 = morph.generate("gel", ["Verb", "Prog", "A1sg"])
    assert surface4 == "geliyorum"


def test_spellchecker():
    spell = akana.SpellChecker()
    assert spell.is_correct("kitap") is True
    assert spell.is_correct("ktap") is False

    suggestions = spell.suggest("ktap", max_distance=2, max_suggestions=5)
    assert len(suggestions) > 0
    assert any(s["word"] == "kitap" for s in suggestions)


def test_disambiguation():
    disambiguator = akana.Disambiguator()
    tokens = ["Ali", "güzel", "kitap", "okudu"]
    parses = disambiguator.disambiguate(tokens)

    assert len(parses) == 4
    assert parses[0]["lemma"] == "ali"
    assert parses[1]["primary_pos"] == "Adj"
    assert parses[2]["primary_pos"] == "Noun"
    assert parses[3]["primary_pos"] == "Verb"
    assert parses[3]["analysis"] == "[okumak:Verb] oku:Verb+du:Past+A3sg"


def test_dependency_parser():
    parser = akana.DependencyParser()
    tokens = ["Ali", "güzel", "kitabı", "okudu"]
    conllu = parser.parse_conllu(tokens)

    assert "okudu" in conllu
    assert "root" in conllu
    assert "nsubj" in conllu
    assert "amod" in conllu


def test_document_pipeline():
    doc = akana.analyze(
        "Ak Ana, Türk mitolojisinde deniz tanrıçasıdır. Ali güzel bir kitap okudu."
    )
    assert len(doc.sentences) == 2
    assert "Ak" in doc.sentences[0].tokens
    assert "kitap" in doc.sentences[1].tokens
    assert len(doc.sentences[0].parses) > 0
    assert all(p["analysis"].startswith("[") for p in doc.sentences[1].parses)


def test_readability_analysis():
    sample_text = (
        "Küçük çocuk bahçede neşeyle koşuyordu. Güneş pırıl pırıl parlıyor, "
        "kuşlar ağaçların dallarında cıvıldıyordu. Annesi onu eve çağırdı ancak "
        "çocuk oyuna devam etmek istedi."
    )
    report = akana.analyze_readability(sample_text)
    assert report.statistics.total_sentences >= 2
    assert report.statistics.total_words >= 15
    assert report.kalyoncu_formula1.score > 0
    assert len(report.kalyoncu_formula1.grade_level) > 0
    assert report.atesman.score > 0
    assert len(report.atesman.grade_level) > 0


def test_syllabification():
    assert akana.syllabify("Türkçe") == ["Türk", "çe"]
    assert akana.syllabify("araba") == ["a", "ra", "ba"]
    assert akana.hyphenate("bilgisayar") == "bil-gi-sa-yar"
    assert akana.count_syllables("öğretmenlerimiz") == 6


def test_number_conversion():
    assert akana.number_to_words(1923) == "bin dokuz yüz yirmi üç"
    assert akana.number_to_words(105) == "yüz beş"
    assert akana.number_to_words(0) == "sıfır"
    assert akana.ordinal_to_words(1) == "birinci"
    assert akana.ordinal_to_words(4) == "dördüncü"
    assert akana.currency_to_words(1250.50, "TL") == "bin iki yüz elli lira elli kuruş"
    assert akana.words_to_number("bin dokuz yüz yirmi üç") == 1923


def test_stemmer_and_stopwords():
    assert akana.stem("kitaplarımızda") in ["kitap", "kitab"]
    assert akana.is_stopword("ve") is True
    assert akana.is_stopword("bilgisayar") is False
    filtered = akana.remove_stopwords(["bu", "güzel", "kitap", "ve", "defter"])
    assert "ve" not in filtered
    assert "bu" not in filtered
    assert "kitap" in filtered


def test_named_entity_recognition():
    text = "Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde İstanbul Üniversitesi bünyesinde 500 TL ödeme yaptı."
    entities = akana.extract_entities(text)
    assert len(entities) >= 3
    labels = [e.label for e in entities]
    assert "PER" in labels or "DATE" in labels
    assert "MONEY" in labels


def test_keyword_extraction():
    text = "Doğal dil işleme ve morfolojik analiz algoritmaları Türkçe metinlerin çözümlenmesinde büyük rol oynar."
    keywords = akana.extract_keywords(text, top_k=5)
    assert len(keywords) > 0
    assert "keyword" in keywords[0]
    assert "score" in keywords[0]


def test_summarization():
    text = (
        "Türkiye Cumhuriyeti 1923 yılında kuruldu. Başkenti Ankara'dır. "
        "Türkçe, Türk dilleri ailesine ait sondan eklemeli zengin bir dildir. "
        "Günümüzde milyonlarca insan tarafından konuşulmaktadır."
    )
    summary = akana.summarize(text, max_sentences=2)
    assert len(summary) == 2


def test_ai_style_auditor():
    text = (
        "Yapay zeka teknolojileri günümüzde kritik bir rol oynamaktadır. "
        "Bu bağlamda; aynı zamanda sadece büyük şirketler için değil, aynı zamanda herkes için vazgeçilmez bir unsurdur. "
        "Peki bu neden böyledir?"
    )
    report = akana.audit_ai_style(text)
    assert report.ai_score > 40.0
    assert "AI" in report.verdict
    assert len(report.findings) > 0

    prompt = akana.humanize_prompt(text, register="akademik")
    assert "Doğallaştırma" in prompt or "Humanizer" in prompt
    assert "Hedef Register" in prompt


def test_syntactic_morphology():
    parses = akana.syntactic_analyze("geldiğimizde")
    assert len(parses) > 0
    parse = next((p for p in parses if p.root == "gel"), None)
    assert parse is not None
    assert parse.root_pos == "VB"
    assert len(parse.inflectional_groups) >= 2
    assert parse.inflectional_groups[0].pos == "VB"
    assert parse.inflectional_groups[1].pos == "NOMP"
    assert parse.inflectional_groups[1].derivation == "PastNom"
    assert parse.inflectional_groups[1].features.get("Case") == "Loc"
    # UD features: the possessor is Number[psor] and Person[psor].
    assert parse.inflectional_groups[1].features == {
        "Case": "Loc",
        "Number": "Sing",
        "Number[psor]": "Plur",
        "Person": "3",
        "Person[psor]": "1",
        "VerbForm": "Vnoun",
    }

    # Cross-categorization check (güzel as ADJ and ADV)
    guzel_parses = akana.syntactic_analyze("güzel")
    pos_set = {p.root_pos for p in guzel_parses}
    assert "ADJ" in pos_set
    assert "ADV" in pos_set

    # Voice, the copula and converbs open groups of their own.
    derivations = {
        word: [
            [ig.derivation for ig in p.inflectional_groups]
            for p in akana.syntactic_analyze(word)
        ]
        for word in ["yaptırıldı", "evdeyim", "gelince"]
    }
    assert [None, "Caus", "Pass"] in derivations["yaptırıldı"]
    assert [None, "Cop"] in derivations["evdeyim"]
    assert [None, "When"] in derivations["gelince"]
    assert akana.syntactic_analyze("xqzzt") == []

    natural_human_text = (
        "Dün sabah erkenden uyandım. Hava çok soğuktu. "
        "Hızlıca giyinip dışarı çıktım çünkü vapuru kaçırmak istemiyordum. "
        "İskelede sıcak bir çay içtim."
    )
    human_report = akana.audit_ai_style(natural_human_text)
    assert human_report.ai_score < 25.0
    assert human_report.verdict == "Doğal İnsan Metni"


def test_humanize_prompt():
    synthetic_ai_text = "Bu doğrultuda eğitim sistemleri hayati bir önem taşımaktadır."
    prompt = akana.humanize_prompt(synthetic_ai_text, register="blog")
    assert "Humanizer" in prompt
    assert "İşlenecek Metin" in prompt


def test_embeddings():
    # Single embedding
    vec = akana.embed("Türkçe doğal dil işleme")
    assert len(vec) == 256
    assert isinstance(vec[0], float)

    # Batch embedding
    texts = ["Merhaba dünya", "Türkiye'nin başkenti Ankara'dır", "Yapay zeka"]
    vecs = akana.embed_batch(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 256

    # Similarity tests
    sim_identical = akana.similarity("kitap", "kitap")
    assert abs(sim_identical - 1.0) < 0.01

    sim_related = akana.similarity("ev", "evler")
    sim_unrelated = akana.similarity("ev", "araba")
    assert sim_related > sim_unrelated

    # Similarity vectors
    v1 = akana.embed("ev")
    v2 = akana.embed("evler")
    sim_vec = akana.similarity_vectors(v1, v2)
    assert abs(sim_vec - sim_related) < 1e-4

    # Embeddings class
    emb = akana.Embeddings()
    assert emb.dimension == 256
    assert len(emb.embed("test")) == 256
    assert emb.similarity("kedi", "kedi") > 0.99


def test_encoder_info_and_check():
    info = akana.encoder_info()
    assert info["sparse_spec"].split(".")[0] == "1"
    assert info["hybrid_layout"] == 1
    assert len(info["fingerprint"]) == 32
    assert len(info["analysis_fingerprint"]) == 32
    assert set(info["resources"]) >= {"tokenizer.json", "turboquant_weights.npz", "analysis"}
    assert akana.check_encoder(info) == "identical"
    assert akana.check_encoder({**info, "fingerprint": "0"}) == "compatible"
    assert akana.check_encoder({**info, "sparse_spec": "0.9"}) == "rebuild"
    assert akana.check_encoder({}) == "rebuild"
