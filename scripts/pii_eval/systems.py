"""Systems the PII evaluation can score: callables `text -> [(start, end, coarse, label)]`.

`akana` is the shipped TurkishPiiEngine; `akana+ner` adds the persons of akana's trained NER
tagger, which the PII engine does not use. A model or a cascade under development registers a
factory in `SYSTEMS`.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# akana entity label -> coarse class of the datasets (None: a type they do not annotate)
AKANA_COARSE = {
    "AD": "person", "KISI": "person", "ANNE_ADI": "person", "BABA_ADI": "person",
    "ANNE_KIZLIK": "person",
    "ADRES": "address", "KONUM": "address", "DOGUM_YERI": "address", "ISYERI": "address",
    "EMAIL": "email",
    "TEL": "phone",
    "OZEL_URL": "url",
    "DOGUM_TARIHI": "date", "OZEL_TARIH": "date",
    "TCKN": "account", "VKN": "account", "VERGI_NO": "account", "IBAN": "account",
    "KART": "account", "HESAP_NO": "account", "MUSTERI_NO": "account", "PASAPORT": "account",
    "EHLIYET": "account", "PLAKA": "account", "SSN": "account", "SGK_NO": "account",
    "SICIL_NO": "account", "SOZLESME_NO": "account", "POLICE_NO": "account",
    "REFERANS": "account", "KRIPTO_CUZDAN": "account", "IMEI": "account", "SASI_NO": "account",
    "MOTOR_NO": "account", "RUHSAT_NO": "account", "CIHAZ_ID": "account",
    "KULLANICI_ADI": "person", "IP_ADRES": "account", "MAC_ADRES": "account",
    "KREDI_NOTU": "account", "KART_SKT": "secret",
    "SIFRE": "secret", "PIN": "secret", "CVV": "secret", "IMZA": "secret",
}

# akana labels named differently in Saturday Labs' tags
SATURDAY_ALIAS = {"PASAPORT": "PASAPORT_NO", "EHLIYET": "EHLIYET_NO"}

PERSON_LABELS = {"AD", "ANNE_ADI", "BABA_ADI", "ANNE_KIZLIK", "KISI"}


def import_akana(pkg: str | None = None):
    if pkg:
        sys.path.insert(0, pkg)
    import akana
    return akana


def akana_engine(pkg=None, embeddings=False, saturday=False):
    akana = import_akana(pkg)
    engine = akana.TurkishPiiEngine(use_embeddings=True) if embeddings else akana.TurkishPiiEngine()

    def detect(text):
        out = []
        for e in engine.detect(text):
            label = e["label"]
            coarse = SATURDAY_ALIAS.get(label, label) if saturday else AKANA_COARSE.get(label)
            out.append((e["start"], e["end"], coarse, label))
        return out

    return detect


def akana_with_ner(pkg=None, saturday=False):
    """The engine plus the NER tagger's persons."""
    akana = import_akana(pkg)
    engine = akana_engine(pkg, saturday=saturday)

    def detect(text):
        persons = [(e.start, e.end, "AD" if saturday else "person", "AD")
                   for e in akana.extract_entities(text) if e.label == "PER"]
        return engine(text) + persons

    return detect


YTU_COARSE = {
    "KISI_AD_SOYAD": "person", "ADRES": "address", "POSTA_KODU": "address", "EMAIL": "email",
    "TELEFON": "phone", "DOGUM_TARIHI": "date", "TCKN": "account", "PLAKA": "account",
    "IP_ADRES": "account", "KIMLIK_BELGE_NO": "account", "PASAPORT_NO": "account",
    "SURUCU_BELGESI_NO": "account", "IBAN_TR": "account", "KART_NO": "account",
    "HESAP_NO": "account", "SWIFT_BIC": "account", "VKN": "account", "MERSIS_NO": "account",
    "ETTN_EFATURA_ID": "account", "KART_CVV": "secret", "KART_SON_KULLANMA": "secret",
}
YTU_SATURDAY = {
    "KISI_AD_SOYAD": "AD", "TELEFON": "TEL", "ADRES": "ADRES", "EMAIL": "EMAIL", "TCKN": "TCKN",
    "DOGUM_TARIHI": "DOGUM_TARIHI", "PLAKA": "PLAKA", "IP_ADRES": "IP_ADRES",
    "PASAPORT_NO": "PASAPORT_NO", "SURUCU_BELGESI_NO": "EHLIYET_NO", "IBAN_TR": "IBAN",
    "KART_NO": "KART", "KART_CVV": "CVV", "KART_SON_KULLANMA": "KART_SKT", "HESAP_NO": "HESAP_NO",
    "VKN": "VERGI_NO", "SAGLIK_BILGISI": "SAGLIK", "TUZEL_KISI": "ISYERI",
}


def ytu_pii(pkg=None, saturday=False, model="ytu-ce-cosmos/modernbert-tr-pii-ner"):
    """ytu-ce-cosmos/modernbert-tr-pii-ner (Apache-2.0), the open Turkish PII model, as its card
    runs it: a token-classification pipeline with first-subword aggregation. Needs torch and
    transformers (the training environment); uses the GPU when there is one."""
    import torch
    from transformers import pipeline

    gpu = torch.cuda.is_available()
    # half precision and small batches on the GPU: this machine has cut power under heavy GPU load
    nlp = pipeline("token-classification", model=model, aggregation_strategy="first",
                   device=0 if gpu else -1, torch_dtype=torch.float16 if gpu else None)
    mapping = YTU_SATURDAY if saturday else YTU_COARSE
    joiners = set(" .@-_/+()")

    def convert(text, ents):
        # the pipeline returns a value in pieces (an e-mail as "ahmet", ".", "yilmaz", ...): pieces
        # of one label separated by at most two joining characters are one value
        merged = []
        for e in sorted(ents, key=lambda x: x["start"]):
            gap = text[merged[-1][1]:e["start"]] if merged else None
            if merged and merged[-1][3] == e["entity_group"] and len(gap) <= 2 and set(gap) <= joiners:
                merged[-1][1] = e["end"]
            else:
                merged.append([e["start"], e["end"], mapping.get(e["entity_group"]), e["entity_group"]])
        return [tuple(m) for m in merged]

    def detect(text):
        return convert(text, nlp(text))

    def batch(texts):
        return [convert(t, ents) for t, ents in zip(texts, nlp(texts, batch_size=8))]

    detect.batch = batch
    return detect


TAGGER_COARSE = {"PERSON": "person", "USERNAME": "person", "ADDRESS": "address", "PHONE": "phone",
                 "EMAIL": "email", "URL": "url", "ID": "account", "SECRET": "secret", "DATE": "date"}
TAGGER_SATURDAY = {"PERSON": "AD", "USERNAME": "KULLANICI_ADI", "ADDRESS": "ADRES", "PHONE": "TEL", "EMAIL": "EMAIL",
                   "URL": "OZEL_URL", "ID": "TCKN", "SECRET": "SIFRE", "DATE": "DOGUM_TARIHI"}
ROLE_MAILBOXES = {"info", "destek", "iletisim", "kvkk", "muhasebe", "satis", "noreply", "no-reply", "insankaynaklari",
                  "basin", "support", "help", "contact", "sales", "yardim", "musterihizmetleri", "bilgi"}


SPOKEN_DIGITS = {"sıfır": "0", "bir": "1", "iki": "2", "üç": "3", "dört": "4", "beş": "5", "altı": "6", "yedi": "7",
                 "sekiz": "8", "dokuz": "9"}
NUMBER_WORDS = {"on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan", "yüz", "bin"}


def public_value(text: str, label: str) -> bool:
    """A call-centre or service number, or a role mailbox: public, left unmasked by the default policy."""
    if label == "PHONE":
        spoken = re.sub(r"\w+", lambda m: SPOKEN_DIGITS.get(m.group().lower(), m.group()), text)
        if any(w.lower() in NUMBER_WORDS for w in re.findall(r"\w+", spoken)):  # "beş yüz otuz iki": unread
            return False
        digits = re.sub(r"\D", "", spoken)
        digits = re.sub(r"^(0090|90)(?=\d{10}$)", "", digits).lstrip("0")
        return digits.startswith(("444", "850", "800", "900")) or len(digits) <= 4
    if label == "EMAIL":
        return re.split(r"\s*(?:@|\[at\]|\(at\)| at )", text.lower())[0].strip() in ROLE_MAILBOXES
    return False


def tagger(pkg=None, saturday=False, model=None, policy=True):
    """A token classifier trained by scripts/pii_train/train_tagger.py (`model`: its directory)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pii_train"))
    import tagger as tg

    # PII_O_BIAS: the tagger's operating point (tagger.Tagger's o_bias), 0 by default, applied to
    # the labels in PII_BIAS_LABELS ("PERSON+DATE") when set, else to all. PII_BIG: one or more
    # further taggers ('+'-joined directories) with the same vocabulary for a cascade (tagger.Cascade),
    # escalating windows whose uncertainty exceeds PII_THRESHOLD (0.1 by default; below 0, every window)
    bias = float(os.environ.get("PII_O_BIAS", 0))
    labels = tuple(os.environ["PII_BIAS_LABELS"].split("+")) if os.environ.get("PII_BIAS_LABELS") else None
    t = tg.Tagger(model or os.environ["PII_MODEL"], batch_size=16, o_bias=bias, bias_labels=labels)
    if os.environ.get("PII_BIG"):
        bigs = [tg.Tagger(path, batch_size=16) for path in os.environ["PII_BIG"].split("+")]
        t = tg.Cascade(t, bigs, float(os.environ.get("PII_THRESHOLD", 0.1)), o_bias=bias, bias_labels=labels)
    mapping = TAGGER_SATURDAY if saturday else TAGGER_COARSE

    def convert(text, spans):
        return [(s, e, mapping[lab], lab) for s, e, lab in spans if not (policy and public_value(text[s:e], lab))]

    def detect(text):
        return convert(text, t.predict([text])[0])

    def batch(texts):
        return [convert(x, spans) for x, spans in zip(texts, t.predict(texts))]

    detect.batch = batch
    return detect


# akana labels whose detectors are exact enough to trust beside the tagger: validated or format-bound
# identifiers, e-mails, phones, provider-format secrets (confidence >= 0.98; the entropy scanner
# reports less), dates after their trigger words, addresses with street and number
AKANA_STRUCTURED = {"EMAIL", "TEL", "TCKN", "VKN", "VERGI_NO", "IBAN", "KART", "KART_SKT", "CVV", "IP_ADRES",
                    "MAC_ADRES", "IMEI", "PLAKA", "SASI_NO", "PASAPORT", "OZEL_URL", "SIFRE", "PIN",
                    "DOGUM_TARIHI", "OZEL_TARIH", "ADRES"}
TAGGER_LABEL_OF = {"EMAIL": "EMAIL", "TEL": "PHONE", "OZEL_URL": "URL", "SIFRE": "SECRET", "PIN": "SECRET",
                   "CVV": "SECRET", "KART_SKT": "SECRET", "DOGUM_TARIHI": "DATE", "OZEL_TARIH": "DATE", "ADRES": "ADDRESS"}


# akana detectors whose values carry a checksum: their span is exact, so it also repairs a tagger
# span that splits or cuts the value
CHECKSUMMED = {"TCKN", "VKN", "VERGI_NO", "IBAN", "KART", "IMEI"}


def merge_spans(model_spans, rule_spans) -> list[tuple[int, int, str]]:
    """The tagger's spans [(start, end, label)] with akana's [(start, end, label, akana label)]: a
    rule span the tagger has nothing on is added; where both found something the tagger's
    boundaries stand (akana's e-mail and date patterns overrun into the next word), except that a
    checksummed identifier replaces the tagger spans it overlaps by their union."""
    out = [tuple(m) for m in model_spans]
    for s, e, lab, akana_label in sorted(rule_spans):
        hit = [m for m in out if m[0] < e and s < m[1]]
        if not hit:
            out.append((s, e, lab))
        elif akana_label in CHECKSUMMED:
            out = [m for m in out if m not in hit]
            out.append((min([s] + [m[0] for m in hit]), max([e] + [m[1] for m in hit]), lab))
    return sorted(out)


def hybrid(pkg=None, saturday=False, model=None, policy=True, names=False):
    """The tagger with akana's structured detectors (merge_spans). `names` also adds akana's person
    names where the tagger found nothing."""
    akana = import_akana(pkg or os.environ.get("AKANA_PKG"))
    engine = akana.TurkishPiiEngine()
    tag = tagger(saturday=saturday, model=model, policy=False)
    mapping = TAGGER_SATURDAY if saturday else TAGGER_COARSE

    def rules(text):
        out = []
        for e in engine.detect(text):
            lab = e["label"]
            if lab in AKANA_STRUCTURED and not (lab == "SIFRE" and e["confidence"] < 0.98):
                out.append((e["start"], e["end"], TAGGER_LABEL_OF.get(lab, "ID"), lab))
            elif names and lab == "AD":
                out.append((e["start"], e["end"], "PERSON", lab))
        return out

    def batch(texts):
        model_spans = [[(s, e, lab) for s, e, _, lab in spans] for spans in tag.batch(texts)]
        return [[(s, e, mapping[lab], lab) for s, e, lab in merge_spans(m, rules(t))
                 if not (policy and public_value(t[s:e], lab))] for t, m in zip(texts, model_spans)]

    def detect(text):
        return batch([text])[0]

    detect.batch = batch
    return detect


SYSTEMS = {
    "akana": akana_engine,
    "akana+ner": akana_with_ner,
    "ytu": ytu_pii,
    "tagger": tagger,
    "hybrid": hybrid,
    "hybrid+names": lambda **kw: hybrid(names=True, **kw),
}
