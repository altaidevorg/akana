"""Behavioural probes: one capability per block, the same sentence frames with varied values, and
the part that must be masked completely (or nothing, for the negatives).

    python scripts/pii_eval/stress.py [--system akana] [--pkg DIR]
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systems  # noqa: E402

GIVEN = ["Ahmet", "Zeynep", "Mehmet", "Elif", "Mustafa", "Ayşe", "Emre", "Merve", "Burak", "Selin",
         "Hüseyin", "Gülşen", "Oğuz", "Şeyma", "İbrahim", "Çağla"]
SURNAMES = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Öztürk", "Aydın", "Arslan",
            "Doğan", "Kılıç", "Aslan", "Çetin", "Koç", "Kurt", "Özdemir"]
AMBIGUOUS = ["Deniz", "Umut", "Barış", "Gül", "Bahar", "Ümit", "Savaş", "Yağmur", "Kader", "Irmak",
             "Cennet", "Sevgi", "Ilgın", "Toprak", "Çınar", "Ege"]
FOREIGN = ["John Smith", "Maria Garcia", "Hiroshi Tanaka", "Olga Petrova", "Pierre Dubois",
           "Ali Reza Karimi", "Fatima Zahra", "Dmitri Ivanov", "Anna Kowalska", "Lukas Müller"]
FRAMES = ["Dün {n} ile görüştüm, yarın tekrar arayacak.", "Kargoyu {n} teslim aldı ama imza atmadı.",
          "Bu konuda {n} haklıydı bence.", "Toplantı notlarını {n} hazırlayacak."]
NEGATIVES = [
    "Deniz kenarında yürüyüş yaptık ve umut dolu bir gün geçirdik.",
    "Barış görüşmeleri yarın Cenevre'de başlayacak.",
    "Kaya tırmanışı için ekipman listesi hazırlandı.",
    "Gül fidanlarını sonbaharda budamak gerekir.",
    "Atatürk Caddesi'ndeki trafik yoğunluğu arttı.",
    "Fatih Sultan Mehmet Köprüsü'nde bakım çalışması var.",
    "Ürün kodu 4829-1123 olan model stokta kalmadı.",  # an order number would be an ID (annotation guideline)
    "Fiyatı 1.250,00 TL olan ürün 3-5 iş gününde teslim edilir.",
    "Kampanya 31.12.2026 tarihine kadar geçerlidir.",
    "Müşteri hizmetlerine 0850 222 33 44 numarasından ulaşabilirsiniz.",
    "Poliçe limitleri dahilinde ödeme yapılacaktır.",
    "Ticaret Sicil Gazetesi'nde ilan edildi.",
    "Kanser araştırmalarına yeni fon ayrıldı.",
    "İslam sanatında geometrik desenler önemlidir.",
    "Sürüm v2.3.1-rc4 yayınlandı, commit a3f9c21 ile birleştirildi.",
    "Python'da user_id ve access_token değişkenlerini tanımla.",
    "Toplantı 14:30'da B-204 numaralı salonda yapılacak.",
    "Yönetmeliğin 12. maddesi 3. fıkrası uyarınca karar verildi.",
    "Ankara'dan İstanbul'a 450 km yol gittik.",
    "Referans olarak geçen yılın raporunu kullandık.",
]


def asciify(s):
    return s.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"))


def tr_lower(s):
    return s.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()


def tckn(rng):
    d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
    d10 = (sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10
    return "".join(map(str, d + [d10, (sum(d) + d10) % 10]))


def run(system, rng):
    def masked(text, value):
        i = text.find(value)
        cov = set()
        for p in system(text):
            cov.update(range(p[0], p[1]))
        return all(k in cov for k in range(i, i + len(value)))

    results = []

    def names(label, transform, pool=None, frames=FRAMES):
        ok = total = 0
        for _ in range(40):
            n = rng.choice(pool) if pool else f"{rng.choice(GIVEN)} {rng.choice(SURNAMES)}"
            shown, must = (lambda v: v if isinstance(v, tuple) else (v, v))(transform(n))
            text = rng.choice(frames).format(n=shown)
            if text.startswith(shown):
                continue
            ok += masked(text, must)
            total += 1
        results.append((label, ok, total))

    def block(label, cases):
        results.append((label, sum(masked(t, v) for t, v in cases), len(cases)))

    suffixed = ["Dosya {n} masasında duruyor.", "Bu karar {n} lehine çıktı."]
    names("full name, Title Case", lambda n: n)
    names("full name, lower case", tr_lower)
    names("full name, CAPITALS", tr_upper)
    names("full name, without Turkish letters", asciify)
    names("full name, lower case without Turkish letters", lambda n: tr_lower(asciify(n)))
    names("given name alone, mid-sentence", lambda n: n.split()[0])
    names("surname + Bey", lambda n: (n.split()[1] + " Bey", n.split()[1]))
    names("foreign full name", lambda n: n, pool=FOREIGN)
    names("ambiguous given name + surname", lambda n: f"{n} {rng.choice(SURNAMES)}", pool=AMBIGUOUS)
    names("ambiguous given name alone, mid-sentence", lambda n: n, pool=AMBIGUOUS)
    names("full name + apostrophe suffix", lambda n: (n + "'ın", n), frames=suffixed)
    names("full name + suffix without apostrophe", lambda n: n + "ın", frames=suffixed)
    names("initial + surname", lambda n: n.split()[0][0] + ". " + n.split()[1])
    names("given + middle initial + surname", lambda n: f"{n.split()[0]} {rng.choice('ABCDEFGHKLMNPRST')}. {n.split()[1]}")

    ids = [tckn(rng) for _ in range(30)]
    block("TCKN plain, no trigger", [(f"Başvuru {t} ile yapıldı.", t) for t in ids])
    block("TCKN spaced 3-3-3-2", [(f"Kimlik: {t[:3]} {t[3:6]} {t[6:9]} {t[9:]} olarak geçti.", f"{t[:3]} {t[3:6]} {t[6:9]} {t[9:]}") for t in ids])
    block("TCKN with a zero-width space", [(f"TC {t[:5]}​{t[5:]} kayıtlı.", f"{t[:5]}​{t[5:]}") for t in ids])
    block("TCKN dotted 3.3.3.2", [(f"Kimlik no {t[:3]}.{t[3:6]}.{t[6:9]}.{t[9:]} olarak düzeltildi.", f"{t[:3]}.{t[3:6]}.{t[6:9]}.{t[9:]}") for t in ids])
    phones = [f"05{rng.randint(30, 59)}{rng.randint(1000000, 9999999)}" for _ in range(30)]
    block("mobile 05xx 3-3-2-2", [(f"Beni {p[:4]} {p[4:7]} {p[7:9]} {p[9:]} numarasından arayın.", f"{p[:4]} {p[4:7]} {p[7:9]} {p[9:]}") for p in phones])
    block("mobile with dots", [(f"İletişim {p[:4]}.{p[4:7]}.{p[7:9]}.{p[9:]} hattı.", f"{p[:4]}.{p[4:7]}.{p[7:9]}.{p[9:]}") for p in phones])
    block("mobile, +90 (5xx) layout", [(f"Tel: +90 ({p[1:4]}) {p[4:7]} {p[7:9]} {p[9:]}", f"+90 ({p[1:4]}) {p[4:7]} {p[7:9]} {p[9:]}") for p in phones])
    dates = [f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1950, 2005)}" for _ in range(30)]
    block("birth date after 'doğum tarihim'", [(f"Doğum tarihim {d} olarak yazılmış.", d) for d in dates])
    block("appointment date and time, no trigger phrase", [(f"Randevu {d} 14:30 olarak ayarlandı.", f"{d} 14:30") for d in dates])
    block("date of a first-person event", [(f"{d} tarihinde ameliyat oldum.", d) for d in dates])
    mails = [f"{asciify(tr_lower(rng.choice(GIVEN)))}.{asciify(tr_lower(rng.choice(SURNAMES)))}{rng.randint(1, 99)}@gmail.com" for _ in range(30)]
    block("e-mail", [(f"Bana {m} adresinden yazın.", m) for m in mails])
    block("e-mail before a sentence end", [(f"Bana {m}. Teşekkürler.", m) for m in mails])
    addresses = ["Atatürk Mah. Gül Sk. No:12 D:3 Kadıköy/İstanbul", "Yeni Mahalle 1203. Sokak No 5 Çankaya Ankara",
                 "Cumhuriyet Mh. İnönü Cd. No:45/7 Bornova İzmir", "Fatih Sultan Mehmet Bulvarı No:8 Kat 2 Nilüfer Bursa"]
    block("postal address", [(f"Adresim {a} olarak kayıtlı.", a) for a in addresses])
    block("postal address, lower case, no punctuation", [(f"adresim {tr_lower(a).replace('.', '').replace(':', ' ')} olarak kayıtlı", tr_lower(a).replace('.', '').replace(':', ' ')) for a in addresses])
    clean = sum(not system(t) for t in NEGATIVES)
    results.append(("PII-free sentences left untouched", clean, len(NEGATIVES)))
    false = [(t, [(p[3], t[p[0]:p[1]]) for p in system(t)]) for t in NEGATIVES if system(t)]
    return results, false


def luhn(rng, prefix="4543", n=16):
    d = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(n - len(prefix) - 1)]
    s = sum(sum(divmod(2 * x, 10)) if i % 2 == 0 else x for i, x in enumerate(reversed(d)))
    return "".join(map(str, d + [(10 - s % 10) % 10]))


def iban(rng):
    bban = f"{rng.randint(0, 99999):05d}0{rng.randint(0, 10**16 - 1):016d}"
    check = 98 - int(bban + "2927" + "00") % 97
    return f"TR{check:02d}{bban}"


SPOKEN = {0: "sıfır", 1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi", 8: "sekiz", 9: "dokuz"}
MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
FILLER = ("Bu yazı, şirketin genel çalışma esaslarını, iş akışlarını ve birimler arasındaki koordinasyonu "
          "anlatan uzun bir giriş bölümüdür; özel bir kişiye ait bilgi içermez. ")
HARD_NEGATIVES = [
    "GET /api/v2/urunler?sayfa=3&kategori=elektronik 200 OK 2025-03-14T10:22:01Z",
    "2025/14 sayılı genelge 12.03.2025 tarihinde yayımlandı.",
    "Resmî Gazete'nin 15.07.2024 tarihli ve 32603 sayılı nüshasında yayımlandı.",
    "Hata kodu E-4021: bağlantı zaman aşımına uğradı (sunucu api.ornek.com.tr).",
    "Paket sürümü 4.2.1, derleme numarası 20250312.7 olarak etiketlendi.",
    "Müşteri hizmetleri: 444 0 123, e-posta: destek@firma.com.tr",
    "İhale kayıt numarası 2025/123456 olan ihalenin ilanı 10.02.2025'te yayımlandı.",
    "Ürün kodu SKU-88213-TR, barkod 8690000123456, ISBN 978-605-09-1234-5.",
    "Ayrıntılar: https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=6698&MevzuatTur=1",
    "SHA256 özeti 3f786850e387550fdab836ed7e6dc881de23001b ile doğrulandı.",
    "Bu yazı 12 Mart 2025'te güncellendi, okuma süresi 5 dakika.",
    "29 Ekim 1923'te Cumhuriyet ilan edildi.",
    "Kampanya 15.06.2026 tarihine kadar tüm şubelerde geçerlidir.",
    "Duyuru No: DUY-2025/0412 — kütüphane 1 Mayıs'ta kapalı olacaktır.",
    "level=warn service=odeme-api status=503 retry_after=30 region=eu-central-1",
    '{"version": "3.1.0", "released": "2025-01-20", "docs": "https://docs.ornek.com.tr/v3?lang=tr"}',
]


def run_hard(system, rng):
    """Harder probes: evasion and layout variants, code-switching, long context, public look-alikes."""
    def masked(text, value):
        i = text.find(value)
        cov = set()
        for p in system(text):
            cov.update(range(p[0], p[1]))
        return i >= 0 and all(k in cov for k in range(i, i + len(value)))

    results = []

    def block(label, cases):
        results.append((label, sum(masked(t, v) for t, v in cases), len(cases)))

    people = [(rng.choice(GIVEN), rng.choice(SURNAMES)) for _ in range(30)]
    trios = zip(people, people[1:] + people[:1], people[2:] + people[:2])
    block("name in an English sentence", [(f"Please forward the contract to {g} {s} before Friday.", f"{g} {s}") for g, s in people])
    block("court header, capitals after a role", [(f"DAVACI : {tr_upper(g)} {tr_upper(s)}\nVEKİLİ : Av. X", f"{tr_upper(g)} {tr_upper(s)}") for g, s in people])
    block("name after a title (Dr., Av., Prof. Dr.)", [(f"Raporu {rng.choice(['Dr.', 'Av.', 'Prof. Dr.', 'Uzm. Dr.'])} {g} {s} imzaladı.", f"{g} {s}") for g, s in people])
    block("name next to an emoji", [(f"Bugün {g} {s}\U0001F389 ile kutladık!", f"{g} {s}") for g, s in people])
    block("name in a form across lines", [(f"Ad:\n{g}\nSoyad:\n{s}\nİmza:", s) for g, s in people])
    block("three names in a list", [(f"Katılımcılar: {a[0]} {a[1]}, {b[0]} {b[1]} ve {c[0]} {c[1]}.", f"{c[0]} {c[1]}") for a, b, c in trios])
    block("chat style, lower case, glued suffix", [(f"yarın {tr_lower(g)}le buluşcam sonra ararım", f"{tr_lower(g)}le") for g, _ in people])
    block("name after 1,500 characters of text", [(FILLER * 9 + f"Dosyayı {g} {s} teslim aldı.", f"{g} {s}") for g, s in people])
    block("zero-width space inside a name", [(f"Dün {g[:2]}​{g[2:]} {s} ile görüştüm.", f"{g[:2]}​{g[2:]} {s}") for g, s in people])
    handles = [f"{asciify(tr_lower(g))}_{asciify(tr_lower(s))}{rng.randint(1, 99)}" for g, s in people]
    block("social handle", [(f"@{h} bunu gördün mü?", h) for h in handles])
    users = [f"{asciify(tr_lower(g))}.{asciify(tr_lower(s))}" for g, s in people]
    block("e-mail obfuscated with [at] and [dot]", [(f"Bana {u} [at] gmail [dot] com adresinden ulaşın.", f"{u} [at] gmail [dot] com") for u in users])
    block("e-mail spelled with 'at' and 'nokta'", [(f"mailim {u} at gmail nokta com", f"{u} at gmail nokta com") for u in users])
    phones = [f"05{rng.randint(30, 59)}{rng.randint(1000000, 9999999)}" for _ in range(30)]
    block("mobile with a space between every digit", [(f"numaram {' '.join(p)} yazın", " ".join(p)) for p in phones])
    block("mobile spoken digit by digit", [(f"numaram {' '.join(SPOKEN[int(c)] for c in p)} arayın", " ".join(SPOKEN[int(c)] for c in p)) for p in phones])
    block("landline with area code", [(f"Evi (0{rng.choice([212, 216, 312, 232])}) {p[4:7]} {p[7:9]} {p[9:]} numarasından arayın.", f"{p[4:7]} {p[7:9]} {p[9:]}") for p in phones])
    block("foreign mobile +49", [(f"Almanya'daki numarası +49 151 {p[3:7]} {p[7:]}.", f"+49 151 {p[3:7]} {p[7:]}") for p in phones])
    ibans = [iban(rng) for _ in range(30)]
    grouped = [" ".join(x[i:i + 4] for i in range(0, 26, 4)) for x in ibans]
    block("IBAN in groups of four", [(f"IBAN: {g} hesabına yatırın.", g) for g in grouped])
    block("IBAN lower case, no spaces", [(f"iban {x.lower()} olarak güncellendi", x.lower()) for x in ibans])
    cards = [luhn(rng) for _ in range(30)]
    block("card number with dashes", [(f"Kart: {c[:4]}-{c[4:8]}-{c[8:12]}-{c[12:]} ile ödendi.", f"{c[:4]}-{c[4:8]}-{c[8:12]}-{c[12:]}") for c in cards])
    block("card last four digits", [(f"Kartımın son dört hanesi {c[-4:]} olan kartla ödedim.", c[-4:]) for c in cards])
    plates = [f"{rng.randint(1, 81):02d} {''.join(rng.choice('ABCDEFGHJKLMNPRSTUVYZ') for _ in range(rng.randint(1, 3)))} {rng.randint(10, 9999)}" for _ in range(30)]
    block("plate, spaced", [(f"Aracın plakası {x} olarak tespit edildi.", x) for x in plates])
    block("plate, lower case, no spaces", [(f"plaka {x.replace(' ', '').lower()} park ihlali", x.replace(" ", "").lower()) for x in plates])
    words = [f"{rng.choice(['Kedi', 'Deniz', 'Mavi', 'Yaz'])}{rng.randint(1000, 9999)}!" for _ in range(30)]
    block("password after 'şifrem'", [(f"Şifrem {w} bunu kimseyle paylaşma.", w) for w in words])
    block("one-time code in an SMS", [(f"Doğrulama kodunuz {c[:6]}. Kimseyle paylaşmayın.", c[:6]) for c in cards])
    days = [f"{rng.randint(1, 28)} {rng.choice(MONTHS)}" for _ in range(30)]
    block("birthday, day and month only", [(f"Doğum günüm {d}, unutma!", d) for d in days])
    flagged = [(t, [(p[3], t[p[0]:p[1]]) for p in system(t)]) for t in HARD_NEGATIVES]
    false = [(t, f) for t, f in flagged if f]
    results.append(("public look-alikes left untouched", len(HARD_NEGATIVES) - len(false), len(HARD_NEGATIVES)))
    return results, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="akana", choices=sorted(systems.SYSTEMS))
    ap.add_argument("--pkg")
    ap.add_argument("--hard", action="store_true", help="the harder probes of run_hard")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    results, false = (run_hard if args.hard else run)(systems.SYSTEMS[args.system](pkg=args.pkg), random.Random(11))
    for label, ok, total in results:
        print(f"{label:50s} {ok:3d}/{total:3d}  {ok / total:6.1%}")
    for text, found in false:
        print(f"  false alarm: {text} -> {found}")


if __name__ == "__main__":
    main()
