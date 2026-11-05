"""Value generators for synthetic Turkish PII: names in every case and spelling with suffixes in
vowel harmony, phones in every layout, e-mails, identifiers with valid (or deliberately broken)
check digits, UAVT addresses, dates and times, URLs, secrets and handles.

Every generator takes a `random.Random` and returns the value's text (and, where the type needs
one, its attributes), so a template can place it and record its span.
"""
from __future__ import annotations

import csv
import random
import re
import string
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PII_SRC = ROOT / "crates" / "akana-core" / "src" / "pii"

VOWELS = "aeıioöuü"
BACK = set("aıou")
ROUNDED = set("oöuü")
VOICELESS = set("fstkçşhp")


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


def tr_title(s: str) -> str:
    return " ".join(w[:1].replace("i", "İ").replace("ı", "I").upper() + tr_lower(w[1:]) if w else w for w in s.split(" "))


def asciify(s: str) -> str:
    return s.translate(str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU"))


# ---------------------------------------------------------------- suffixes in vowel harmony

def _last_vowel(word: str) -> str:
    for c in reversed(tr_lower(word)):
        if c in VOWELS:
            return c
    return "e"


def suffix(word: str, case: str) -> str:
    """The case suffix of `word` (without apostrophe): gen, dat, acc, loc, abl, ins, pl."""
    w = tr_lower(word)
    v = _last_vowel(w)
    back, rounded = v in BACK, v in ROUNDED
    a = "a" if back else "e"
    i = ("u" if rounded else "ı") if back else ("ü" if rounded else "i")
    ends_vowel = w[-1:] in VOWELS
    d = "t" if w[-1:] in VOICELESS else "d"
    return {
        "gen": ("n" if ends_vowel else "") + i + "n",
        "dat": ("y" if ends_vowel else "") + a,
        "acc": ("y" if ends_vowel else "") + i,
        "loc": d + a,
        "abl": d + a + "n",
        "ins": ("y" if ends_vowel else "") + "l" + a,
        "pl": "l" + a + "r",
    }[case]


def inflect(value: str, case: str | None, apostrophe: str | None = "'") -> tuple[str, str]:
    """`value` with a case suffix: (written form, the part to mask). With an apostrophe the suffix
    stays outside the mask; glued on (informal text) it is part of the masked word."""
    if not case:
        return value, value
    suf = suffix(value.split()[-1], case)
    if value.isupper():
        suf = tr_upper(suf)
    if apostrophe:
        return f"{value}{apostrophe}{suf}", value
    return value + suf, value + suf


# ---------------------------------------------------------------- names

def _rust_list(path: Path, name: str) -> list[str]:
    src = path.read_text(encoding="utf-8")
    m = re.search(name + r"[^=]*=\s*(?:\{[^\[]*)?&?\[(.*?)\]", src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


SURNAME_HEADS = ["ak", "kara", "öz", "er", "gök", "yıl", "demir", "çelik", "ay", "gün", "al", "el", "taş",
                 "tek", "kılıç", "boz", "ulu", "sarı", "uz", "dur", "kut", "oğuz", "tun", "yüce", "bal",
                 "kaya", "deniz", "doğan", "arslan", "şahin", "kurt", "koç", "yalın", "toprak", "yıldız",
                 "altın", "gümüş", "ateş", "turan", "uçar", "keskin", "erdem", "aydın", "güven", "bilge"]
SURNAME_TAILS = ["soy", "kaya", "taş", "oğlu", "türk", "han", "can", "er", "dağ", "el", "kan", "ay",
                 "ok", "gül", "su", "yol", "baş", "göz", "yürek", "bilek", "kol", "dere", "tepe", "ova",
                 "men", "sel", "tekin", "alp", "oğuz", "demir", "kurt", "ateş", "ışık", "yiğit", "budak"]


@lru_cache(maxsize=1)
def name_lists() -> dict:
    given = {tr_lower(w.strip()) for w in open(PII_SRC / "person_names.txt", encoding="utf-8")
             if w.strip() and not w.startswith("#")}
    given |= set(_rust_list(PII_SRC / "names_data.rs", "COMMON_GIVEN_NAMES"))
    common_given = _rust_list(PII_SRC / "names_data.rs", "COMMON_GIVEN_NAMES")
    surnames = set(_rust_list(PII_SRC / "names_data.rs", "COMMON_SURNAMES"))
    # surnames of WikiNER's training persons: the last word of a person whose first word is a
    # known given name
    wikiner = ROOT / ".cache" / "ner" / "wikiner" / "train.tsv"
    if wikiner.is_file():
        tokens, tags = [], []
        for line in open(wikiner, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            t, g = line.split("\t")
            tokens.append(t)
            tags.append(g)
        i = 0
        while i < len(tokens):
            if tags[i] == "B-PER":
                j = i + 1
                while j < len(tokens) and tags[j] == "I-PER":
                    j += 1
                words = [w.split("'")[0].split("’")[0] for w in tokens[i:j]]
                if len(words) >= 2 and tr_lower(words[0]) in given and words[-1].isalpha() and len(words[-1]) > 2:
                    surnames.add(tr_lower(words[-1]))
                i = j
            else:
                i += 1
    foreign_given = _rust_list(PII_SRC / "names_data.rs", "INTERNATIONAL_GIVEN_NAMES")
    ambiguous = _rust_list(PII_SRC / "names_data.rs", "AMBIGUOUS_NAMES")
    foreign_surnames = ["smith", "johnson", "williams", "brown", "garcia", "miller", "müller", "schmidt",
                        "schneider", "fischer", "dubois", "martin", "bernard", "rossi", "russo", "ivanov",
                        "petrov", "tanaka", "suzuki", "kim", "lee", "park", "nguyen", "kowalski", "novak",
                        "horvath", "papadopoulos", "andersson", "nielsen", "silva", "santos", "fernandez"]
    return {"given": sorted(given), "common_given": common_given, "surnames": sorted(surnames),
            "foreign_given": foreign_given, "foreign_surnames": foreign_surnames, "ambiguous": ambiguous}


def surname(rng: random.Random) -> str:
    lists = name_lists()
    if rng.random() < 0.25:
        head, tail = rng.choice(SURNAME_HEADS), rng.choice(SURNAME_TAILS)
        return head + tail if head[-1:] != tail[:1] else head + tail[1:]
    return rng.choice(lists["surnames"])


def given_name(rng: random.Random, common=0.5) -> str:
    lists = name_lists()
    return rng.choice(lists["common_given"] if rng.random() < common else lists["given"])


def person(rng: random.Random, foreign=0.08, ambiguous=0.08) -> str:
    """A person's name in one of the shapes and cases people write."""
    lists = name_lists()
    if rng.random() < foreign:
        g, s = rng.choice(lists["foreign_given"]), rng.choice(lists["foreign_surnames"])
    else:
        g = rng.choice(lists["ambiguous"]) if rng.random() < ambiguous else given_name(rng)
        s = surname(rng)
    shape = rng.random()
    if shape < 0.55:
        parts = [g, s]
    elif shape < 0.65:
        parts = [g, given_name(rng), s]
    elif shape < 0.75:
        parts = [g]
    elif shape < 0.80:
        parts = [g, f"{rng.choice('ABCDEFGHİKLMNOPRSTUVYZ')}.", s]
    elif shape < 0.85:
        parts = [f"{tr_upper(g[0])}.", s]
    elif shape < 0.90:
        parts = [g, f"{s}-{surname(rng)}" if rng.random() < 0.5 else f"{s} {surname(rng)}"]
    else:
        parts = [s]
    case = rng.random()
    if case < 0.62:
        text = " ".join(tr_title(p) if not p.endswith(".") else p for p in parts)
    elif case < 0.78:
        text = tr_lower(" ".join(parts))
    elif case < 0.88:
        text = tr_upper(" ".join(parts))
    else:  # the official form: given names in title case, surname in capitals
        text = " ".join(tr_title(p) for p in parts[:-1]) + (" " if len(parts) > 1 else "") + tr_upper(parts[-1])
    if rng.random() < 0.12:
        text = asciify(text)
    return text.strip()


def handle(rng: random.Random) -> str:
    g, s = asciify(tr_lower(given_name(rng))), asciify(tr_lower(surname(rng)))
    body = rng.choice([f"{g}{s}", f"{g}_{s}", f"{g}.{s}", f"{g}{rng.randint(1, 99)}", f"{g}_{s[:1]}{rng.randint(70, 9999)}",
                       f"{s}{g[:1]}", f"the{g}", f"{g}{s}{rng.randint(1, 9)}"])
    return ("@" if rng.random() < 0.7 else "") + body


# ---------------------------------------------------------------- phones

ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def number_words(group: str) -> str:
    """A digit group read out as people say it: "532" beş yüz otuz iki, "07" sıfır yedi."""
    if group.startswith("0"):
        return " ".join(["sıfır"] + ([number_words(group[1:])] if group[1:] else []))
    n = int(group)
    h, t, o = n // 100, n // 10 % 10, n % 10
    parts = ([("" if h == 1 else ONES[h] + " ") + "yüz"] if h else []) + ([TENS[t]] if t else []) + ([ONES[o]] if o else [])
    return " ".join(parts)


def spoken_phone(rng: random.Random) -> str:
    """A mobile number read out, as in call transcripts and voice requests."""
    area, sub = f"5{rng.randint(0, 5)}{rng.randint(0, 9)}", f"{rng.randint(100, 999)}{rng.randint(1000, 9999)}"
    groups = ["0" + area, sub[:3], sub[3:5], sub[5:]]
    style = rng.random()
    if style < 0.4:  # digit by digit
        return " ".join(ONES[int(c)] or "sıfır" for c in "".join(groups))
    sep = rng.choice([" ", ", ", " "])
    if style < 0.8:  # grouped: sıfır beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi
        return sep.join(number_words(g) for g in groups)
    return f"0{area} " + sep.join(number_words(g) for g in groups[1:])  # area code in digits


def phone(rng: random.Random, kind: str = "mobile") -> str:
    """kind: mobile, landline, callcentre (444 / 0850 / 0800: a public number), foreign, malformed, spoken."""
    if kind == "spoken":
        return spoken_phone(rng)
    if kind == "callcentre":
        return rng.choice([f"444 {rng.randint(0, 9)} {rng.randint(100, 999)}", f"444 {rng.randint(1000, 9999)}",
                           f"0850 {rng.randint(200, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
                           f"0800 {rng.randint(200, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}"])
    if kind == "foreign":
        return rng.choice([f"+44 20 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
                           f"+49 30 {rng.randint(1000000, 9999999)}", f"+1 ({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
                           f"+33 6 {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}"])
    if kind == "landline":
        area = rng.choice(["212", "216", "312", "232", "224", "242", "322", "262", "332", "352", "412", "462"])
    else:
        area = f"5{rng.randint(0, 5)}{rng.randint(0, 9)}"
    sub = f"{rng.randint(100, 999)}{rng.randint(1000, 9999)}"
    if kind == "malformed":
        sub += str(rng.randint(0, 9))
    a, b, c, d = area, sub[:3], sub[3:5], sub[5:]
    layouts = [
        f"0{a} {b} {c} {d}", f"0{a} {b} {c}{d}", f"0{a}{b}{c}{d}", f"0({a}) {b} {c} {d}", f"({a}) {b}-{c}-{d}",
        f"+90 {a} {b} {c} {d}", f"+90 ({a}) {b} {c} {d}", f"+90{a}{b}{c}{d}", f"90{a}{b}{c}{d}",
        f"0{a}-{b}-{c}-{d}", f"0{a}.{b}.{c}.{d}", f"{a} {b} {c} {d}", f"0 {a} {b} {c} {d}", f"0090 {a} {b} {c} {d}",
        f"+90 {a} {b}{c}{d}", f"0{a} {b} {c}{d}"[:-1] + d[-1],
    ]
    return rng.choice(layouts)


# ---------------------------------------------------------------- e-mails, URLs, handles

DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com", "yandex.com", "mynet.com",
           "windowslive.com", "proton.me", "gmail.com.tr", "superonline.net"]


def company_domain(rng: random.Random) -> str:
    word = asciify(tr_lower(rng.choice(["anadolu", "marmara", "ege", "yildiz", "deniz", "kartal", "atlas", "toros",
                                        "bosfor", "kervan", "mavi", "akdeniz", "safir", "zirve", "orkide", "ekin"])))
    tail = rng.choice(["lojistik", "yazilim", "insaat", "sigorta", "danismanlik", "tekstil", "gida", "enerji", "medikal", "otomotiv"])
    return f"{word}{tail}.{rng.choice(['com', 'com.tr', 'net', 'net.tr', 'org.tr', 'gen.tr', 'av.tr', 'k12.tr', 'edu.tr'])}"


def email(rng: random.Random, kind: str = "personal") -> str:
    """kind: personal, work, role (a public mailbox), obfuscated, spoken."""
    if kind == "role":
        return f"{rng.choice(['info', 'destek', 'iletisim', 'kvkk', 'muhasebe', 'satis', 'noreply', 'insankaynaklari', 'basin'])}@{company_domain(rng)}"
    g, s = asciify(tr_lower(given_name(rng))), asciify(tr_lower(surname(rng)))
    local = rng.choice([f"{g}.{s}", f"{g}{s}", f"{g}_{s}", f"{g[0]}{s}", f"{g}.{s}{rng.randint(1, 99)}",
                        f"{g}{rng.randint(1970, 2010)}", f"{s}.{g}", f"{g}-{s}", f"{g}.{s}+{rng.choice(['fatura', 'kargo', 'is'])}"])
    domain = company_domain(rng) if kind == "work" else rng.choice(DOMAINS)
    if kind == "obfuscated":
        host, tld = domain.split(".", 1)
        return rng.choice([f"{local} [at] {host} [dot] {tld}", f"{local}(at){host}.{tld}", f"{local} @ {host} . {tld}",
                           f"{local} at {host} dot {tld}"])
    if kind == "spoken":
        host, tld = domain.split(".", 1)
        return f"{local.replace('.', ' nokta ')} at {host} nokta {tld.replace('.', ' nokta ')}"
    return f"{local}@{domain}"


def public_domain(rng: random.Random) -> str:
    """A public institution's or publisher's domain."""
    word = rng.choice(["ankara", "izmir", "kadikoy", "nilufer", "meram", "tepebasi", "osmangazi", "konak", "seyhan"])
    topic = rng.choice(["kultur", "ulasim", "vergi", "saglik", "egitim", "mevzuat", "duyuru", "kutuphane", "etkinlik", "haber"])
    return rng.choice([f"{word}.bel.tr", f"{topic}{rng.choice(['rehberi', 'bilgi', 'portali', 'merkezi'])}.gov.tr",
                       f"{word}.edu.tr", f"{topic}{rng.choice(['haber', 'gundem', 'arsiv'])}.com.tr", f"{topic}{word}.org.tr",
                       company_domain(rng)])


def url(rng: random.Random, personal: bool = True) -> str:
    host = rng.choice(["portal", "musteri", "online", "app", "hesap", "e-islem", "randevu"]) + "." + company_domain(rng)
    if not personal:  # a public page, also with ordinary query parameters and anchors
        path = rng.choice(["hakkimizda", "iletisim", "kampanyalar", "sss", "urunler/yeni-sezon", "blog/kvkk-rehberi", "duyurular",
                           f"haberler/{rng.randint(2015, 2029)}/kultur-programi", "mevzuat/yonetmelikler", "rehber/basvuru-adimlari",
                           "etkinlikler/takvim", "belgeler/surum-notlari", "yardim/odeme-yontemleri", "arsiv/yayinlar", "ihale-ilanlari"])
        short = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(rng.randint(2, 6)))
        query = rng.choice(["", "", "", f"?sayfa={rng.randint(2, 40)}", f"?lang={rng.choice(['tr', 'en'])}", f"?bolum={short}",
                            f"?kategori={rng.choice(['duyurular', 'haberler', 'etkinlik', 'mevzuat', 'kampanya'])}", f"?id={rng.randint(100, 99999)}",
                            f"?q={rng.choice(['otopark', 'vergi+takvimi', 'nobetci+eczane', 'burs'])}", f"?sekme={short}", f"#{short}",
                            f"?utm_source={rng.choice(['bulten', 'twitter', 'eposta'])}&utm_medium={rng.choice(['email', 'social'])}",
                            f"?yil={rng.randint(2010, 2029)}&donem={rng.randint(1, 4)}", f"?v={short}{rng.randint(10, 99)}"])
        return f"{rng.choice(['https://', 'https://www.', 'http://', 'www.'])}{public_domain(rng)}/{path}{query}"
    token = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(8, 24)))
    rid = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(7))
    return rng.choice([f"https://{host}/basvuru/{rid}/onay/{token[:8].lower()}", f"https://{host}/dosya/{rid}?token={token}",
                       f"https://{host}/reset-password?k={token}", f"https://{host}/u/{asciify(tr_lower(given_name(rng)))}{rng.randint(10, 99)}",
                       f"http://{host}/fatura/goruntule?id={rng.randint(100000, 999999)}&sig={token[:12]}",
                       f"https://drive.{company_domain(rng)}/paylas/{token}"])


# ---------------------------------------------------------------- identifiers

def tckn(rng: random.Random, valid: bool = True) -> str:
    d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
    d10 = (sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10
    d11 = (sum(d) + d10) % 10
    if not valid:
        d11 = (d11 + rng.randint(1, 9)) % 10
    return "".join(map(str, d + [d10, d11]))


def vkn(rng: random.Random) -> str:
    digits = [rng.randint(0, 9) for _ in range(9)]
    total = 0
    for i, dgt in enumerate(digits):
        tmp = (dgt + 10 - (i + 1)) % 10
        if tmp == 9:
            total += tmp
        elif tmp:
            total += (tmp * 2 ** (10 - (i + 1))) % 9 or 9
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def iban_tr(rng: random.Random, valid: bool = True) -> str:
    bban = f"{rng.randint(10, 999):05d}0" + "".join(str(rng.randint(0, 9)) for _ in range(16))
    num = int("".join(str(int(c, 36)) for c in bban + "TR00"))
    check = 98 - num % 97
    if not valid:
        check = (check + rng.randint(1, 50)) % 97 + 2
    raw = f"TR{check:02d}{bban}"
    style = rng.random()
    if style < 0.45:
        return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))
    if style < 0.85:
        return raw
    if style < 0.92:
        return tr_lower(raw[:2]) + raw[2:]
    groups = [raw[i:i + 4] for i in range(0, len(raw), 4)]
    return " ".join(groups[:2] + ["****"] * (len(groups) - 3) + groups[-1:])


def luhn_complete(prefix: str, length: int, rng: random.Random) -> str:
    digits = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(length - len(prefix) - 1)]
    total = 0
    for i, dgt in enumerate(reversed(digits)):
        if i % 2 == 0:
            dgt *= 2
            dgt -= 9 if dgt > 9 else 0
        total += dgt
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def card(rng: random.Random) -> str:
    raw = luhn_complete(rng.choice(["4", "51", "55", "9792", "65", "37"]), 16, rng)
    style = rng.random()
    if style < 0.5:
        return " ".join(raw[i:i + 4] for i in range(0, 16, 4))
    if style < 0.75:
        return raw
    if style < 0.88:
        return "-".join(raw[i:i + 4] for i in range(0, 16, 4))
    return f"{raw[:4]} **** **** {raw[-4:]}"


def plate(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVYZ") for _ in range(rng.randint(1, 3)))
    num = rng.randint(10, 9999) if len(letters) == 1 else rng.randint(10, 999)
    p = f"{rng.randint(1, 81):02d} {letters} {num}"
    return p.replace(" ", "") if rng.random() < 0.2 else p


def record_id(rng: random.Random, subtype: str) -> str:
    """Account, customer, order and ticket numbers in vendor-like formats."""
    up = string.ascii_uppercase
    prefix = {"customer": ["MUS", "MST", "UYE", "ABN", "POL", "SOZ", "CUST", "MBR"],
              "transaction": ["SIP", "TKT", "KRG", "FAT", "BSV", "REF", "ORD", "INV", "TALEP"],
              "financial": ["HSP", "ACC", "KRT"]}.get(subtype, ["ID"])
    p = rng.choice(prefix)
    year = rng.randint(2018, 2029)
    alnum = "".join(rng.choice(up + string.digits) for _ in range(rng.randint(4, 8)))
    return rng.choice([f"{p}-{year}-{rng.randint(1000, 999999)}", f"{p}{rng.randint(100000, 99999999)}",
                       f"{p}-{alnum}", f"{p}/{year}/{alnum}", f"{rng.randint(10000000, 9999999999)}",
                       f"{alnum[:4]}-{rng.randint(100000, 999999)}-{alnum[-2:]}", f"{p}-{rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"])


def document_code(rng: random.Random) -> str:
    """The code or number of a public document, notice, circular, standard, release or page: in the
    same shapes as record_id, and not personal data (annotation_guideline.md, ID)."""
    up = string.ascii_uppercase
    p = rng.choice(["DUY", "GNL", "YON", "KLV", "RHB", "BLT", "ILN", "SRM", "AKD", "BLD", "KRR", "TBL", "DOC", "ANN",
                    "REL", "GZT", "TEB", "PRS", "ETK", "MEV", "STD", "BLG", "YYN", "HBR"])
    year = rng.randint(2000, 2029)
    alnum = "".join(rng.choice(up + string.digits) for _ in range(rng.randint(3, 6)))
    return rng.choice([f"{p}-{year}-{alnum}", f"{p}-{year}-{rng.randint(1, 999):03d}", f"{p}-{alnum}", f"{p}/{year}/{rng.randint(1, 99)}",
                       f"{year}/{rng.randint(1, 60)}", f"{p}{year}{rng.randint(100, 999)}", f"{p}.{year}.{rng.randint(1, 99):02d}",
                       f"TS {rng.randint(100, 13000)}", f"ISO {rng.choice(['27001', '9001', '14001', '22301', '45001'])}:{rng.choice(['2013', '2015', '2018', '2022'])}",
                       f"CVE-{year}-{rng.randint(1000, 49999)}", f"RFC {rng.randint(700, 9800)}"])


def identifier(rng: random.Random) -> tuple[str, dict]:
    kind = rng.random()
    if kind < 0.17:
        return tckn(rng, valid=rng.random() < 0.85), {"subtype": "national"}
    if kind < 0.25:
        v = vkn(rng)
        return v, {"subtype": "national"}
    if kind < 0.40:
        return iban_tr(rng, valid=rng.random() < 0.9), {"subtype": "financial"}
    if kind < 0.50:
        return card(rng), {"subtype": "financial"}
    if kind < 0.56:
        return f"{rng.choice('UASZ')}{rng.randint(10000000, 99999999)}", {"subtype": "national"}
    if kind < 0.63:
        return plate(rng), {"subtype": "vehicle"}
    if kind < 0.68:
        return luhn_complete(str(rng.randint(10, 99)), 15, rng), {"subtype": "device"}
    if kind < 0.72:
        return ".".join(str(rng.randint(1, 254)) for _ in range(4)), {"subtype": "device"}
    if kind < 0.84:
        return record_id(rng, "customer"), {"subtype": "customer"}
    if kind < 0.96:
        return record_id(rng, "transaction"), {"subtype": "transaction"}
    return record_id(rng, "financial"), {"subtype": "financial"}


# ---------------------------------------------------------------- dates and times

MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
MONTH_ABBR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]


def date(rng: random.Random, years=(1950, 2029), with_time=None) -> str:
    y, m, d = rng.randint(*years), rng.randint(1, 12), rng.randint(1, 28)
    hh, mm, ss = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    fmt = rng.random()
    if fmt < 0.25:
        s = f"{d:02d}.{m:02d}.{y}"
    elif fmt < 0.35:
        s = f"{d}.{m}.{y}"
    elif fmt < 0.47:
        s = f"{d:02d}/{m:02d}/{y}"
    elif fmt < 0.52:
        s = f"{d}/{m}/{y}"
    elif fmt < 0.60:
        s = f"{y}-{m:02d}-{d:02d}"
    elif fmt < 0.66:
        s = f"{d:02d}-{m:02d}-{y}"
    elif fmt < 0.82:
        s = f"{d} {MONTHS[m - 1]} {y}"
    elif fmt < 0.86:
        s = f"{d} {MONTH_ABBR[m - 1]} {y}"
    elif fmt < 0.91:
        s = f"{d} {MONTHS[m - 1]}"
    elif fmt < 0.95:
        s = f"{MONTHS[m - 1]} {y}"
    else:
        s = f"{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}" + rng.choice(["", "+03:00", "Z"])
        return s
    if with_time or (with_time is None and rng.random() < 0.18):
        s += rng.choice([f" {hh:02d}:{mm:02d}", f" saat {hh:02d}.{mm:02d}", f" {hh:02d}:{mm:02d}:{ss:02d}", f", {hh:02d}:{mm:02d}"])
    if rng.random() < 0.08:
        s = tr_lower(s)
    return s


# ---------------------------------------------------------------- secrets

def secret(rng: random.Random) -> str:
    kind = rng.random()
    word = asciify(tr_title(rng.choice(["limon", "kartal", "deniz", "galata", "papatya", "elma", "yildiz", "aslan", "bulut", "ruzgar"])))
    if kind < 0.25:
        return rng.choice([f"{word}{rng.randint(10, 9999)}{rng.choice('!._*#?')}", f"{word}{rng.choice('!._*')}{rng.randint(1, 99)}",
                           f"{tr_lower(word)}{rng.randint(1900, 2024)}", f"{word[:3]}{rng.randint(100, 999)}{tr_lower(word[3:])}{rng.choice('!@#')}"])
    if kind < 0.45:
        return str(rng.randint(1000, 999999)).zfill(rng.choice([4, 6]))
    if kind < 0.55:
        return str(rng.randint(100, 999))
    if kind < 0.68:
        return "sk-" + rng.choice(["", "proj-"]) + "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(40))
    if kind < 0.76:
        return "ghp_" + "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(36))
    if kind < 0.82:
        return "AKIA" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    if kind < 0.90:
        part = lambda n: "".join(rng.choice(string.ascii_letters + string.digits + "-_") for _ in range(n))
        return f"eyJ{part(30)}.eyJ{part(40)}.{part(43)}"
    return "".join(rng.choice(string.ascii_letters + string.digits + "!@#$%&*") for _ in range(rng.randint(10, 20)))


# ---------------------------------------------------------------- addresses (UAVT)

UAVT = ROOT / ".cache" / "uavt"
STREET_KIND = {"1": ("Meydanı", "Meydanı", "meydanı"), "2": ("Bulvarı", "Blv.", "bulv"), "3": ("Caddesi", "Cad.", "cd"),
               "4": ("Sokak", "Sk.", "sk"), "5": ("Küme Evler", "Küme Evler", "küme evler"), "0": ("Sokak", "Sk.", "sk")}


@lru_cache(maxsize=1)
def uavt(sample_streets: int = 150_000, seed: int = 7):
    quarters = {}
    with open(UAVT / "city_district_village_quarter.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            quarters[row["QuarterId"]] = (row["CityName"], row["DistrictName"], row["QuarterName"])
    rng = random.Random(seed)
    streets = []
    with open(UAVT / "streets.csv", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for k, row in enumerate(reader):  # reservoir sample of the streets that have a quarter
            if row[1] not in quarters:
                continue
            item = (row[1], row[5], row[7])
            if len(streets) < sample_streets:
                streets.append(item)
            else:
                j = rng.randint(0, k)
                if j < sample_streets:
                    streets[j] = item
    return quarters, streets


def address(rng: random.Random) -> str:
    quarters, streets = uavt()
    qid, street, kind = rng.choice(streets)
    city, district, quarter = quarters[qid]
    full, abbr, low = STREET_KIND.get(kind, STREET_KIND["4"])
    no, flat = rng.randint(1, 250), rng.randint(1, 60)
    style = rng.random()
    street_t, quarter_t, district_t, city_t = tr_title(street), tr_title(quarter), tr_title(district), tr_title(city)
    if style < 0.35:
        s = f"{quarter_t} Mah. {street_t} {abbr} No:{no} D:{flat} {district_t}/{city_t}"
    elif style < 0.55:
        s = f"{quarter_t} Mahallesi {street_t} {full} No: {no} Daire: {flat} {district_t} / {city_t}"
    elif style < 0.65:
        site = tr_title(rng.choice(["Yıldız", "Gül", "Park", "Mavi", "Lale", "Çınar", "Doğa", "Panorama"]))
        s = f"{quarter_t} Mah. {street_t} {abbr} {site} Sitesi {rng.choice('ABCDE')} Blok Kat:{rng.randint(1, 15)} No:{no}/{flat} {district_t}/{city_t}"
    elif style < 0.80:
        s = f"{tr_lower(quarter)} mh {tr_lower(street)} {low} no {no} d {flat} {tr_lower(district)} {tr_lower(city)}"
    elif style < 0.88:
        s = f"{tr_upper(quarter)} MAH. {tr_upper(street)} {tr_upper(abbr)} NO:{no}/{flat} {tr_upper(district)}/{tr_upper(city)}"
    elif style < 0.94:
        s = f"{street_t} {full} No:{no}/{flat}, {quarter_t} Mah., {rng.randint(1, 81):02d}{rng.randint(0, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)} {district_t}/{city_t}"
    else:
        s = f"{quarter_t} Mah. {rng.randint(1000, 3999)}. Sokak No:{no} {district_t}/{city_t}"
    if rng.random() < 0.1:
        s = asciify(s)
    return s
