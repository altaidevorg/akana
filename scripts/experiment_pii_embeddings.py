"""Can embeddings help pattern- and dictionary-based PII detection where the rules cannot decide?

Four cases, each a small labeled set built here (no data download):
  A. a given name that is also a word, at a sentence start with nothing else around it
     (Deniz yarın gelecek / Deniz bugün dalgalı)
  B. a health condition the dictionary lacks, before a clinical trigger (fibromiyalji tanısı),
     against hobbies and faults (futbol hastasıyım, fren arızası teşhisi); a second set is held
     out from every choice made here
  C. a password-like value with no trigger word, where only the context says secret
     (Bunu kimseyle paylaşma: maviBalina7 / Sipariş numaranız maviBalina7)
  D. a private date (randevu, ameliyat, doğum) against a procedural or public one

Each is scored for the engine without and with use_embeddings=True, and for embedding methods
on the same 256-dim Model2Vec vectors: prototype (centroid) similarity, a two-sided prototype
comparison, and a logistic regression (cross-validated, or trained on one set and tested on
another). Prototype words are kept out of the test items, which earlier runs did not do and
which flattered them.

Findings (akana 0.8.0): embeddings help for B (the engine now uses the two-sided
comparison with a morphology guard: 0 -> 0.89 recall on the held-out set), a little for A
(0.61-0.69 accuracy, not reliable), and not for C or D, where explicit trigger words do better.

Usage: python scripts/experiment_pii_embeddings.py
"""
import random
import re
import sys

import numpy as np

import akana

rng = random.Random(11)
E0 = akana.TurkishPiiEngine()
E1 = akana.TurkishPiiEngine(use_embeddings=True)


def emb(text):
    v = np.asarray(akana.embed(text), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def cos(a, b):
    return float(a @ b)


def fit(X, y, l2=1e-2, epochs=400, lr=0.5):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        g = 1 / (1 + np.exp(-(X @ w + b))) - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return lambda Z: (1 / (1 + np.exp(-(Z @ w + b)))) >= 0.5


def cv(X, y, groups=None, folds=5):
    """Predictions of a logistic regression with stratified (or group) k-fold CV."""
    X, y = np.asarray(X), np.asarray(y, dtype=float)
    keys = groups if groups is not None else list(range(len(y)))
    uniq = sorted(set(keys), key=str)
    rng.shuffle(uniq)
    fold = {k: i % folds for i, k in enumerate(uniq)}
    pred = np.zeros(len(y), dtype=bool)
    for f in range(folds):
        tr = [i for i in range(len(y)) if fold[keys[i]] != f]
        te = [i for i in range(len(y)) if fold[keys[i]] == f]
        if te:
            pred[te] = fit(X[tr], y[tr])(X[te])
    return pred


def report(name, y, pred):
    y, pred = np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool)
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    print(f"  {name:46s} recall {tp / max(1, tp + fn):.2f}  precision {prec:.2f}  "
          f"accuracy {(tp + tn) / len(y):.2f}")


# ------------------------------------------------------------------------------------- A
PERSON = [
    "Deniz yarın sabah ofise gelecek.", "Barış dün akşam beni aradı.", "Umut sınavdan yüksek not aldı.",
    "Gül bugün işe geç kaldı.", "Bahar annesine bir mektup yazdı.", "Yağmur telefonunu evde unutmuş.",
    "Derya toplantıda sunum yaptı.", "Ümit yeni bir işe başladı.", "Metin raporu henüz göndermedi.",
    "Emin bu konuda bizimle konuşmak istiyor.", "Zafer üniversiteyi kazandı.", "Güneş kardeşiyle sinemaya gitti.",
    "Kader ödevini teslim etmedi.", "Su doğum gününü arkadaşlarıyla kutladı.", "Nur hastaneden taburcu oldu.",
    "Ada okula yeni başladı.", "Ekin piyano dersine gidiyor.", "Onur bize akşam yemeğine gelecek.",
    "Özlem e-postama cevap vermedi.", "Duygu yarın İzmir'e taşınıyor.", "Sevgi çocuklarını okula bıraktı.",
    "Eylül bana bir hediye aldı.", "Defne arkadaşlarıyla tatile çıktı.", "Irmak ehliyet sınavını geçti.",
    "Pınar müdürle görüşmeye gitti.", "Çağrı şirketten istifa etti.", "Ufuk maaş zammı istedi.",
    "Doğa kahvaltıyı hazırladı.", "Evren bilgisayarını tamir ettirdi.", "Bulut futbol maçında gol attı.",
    "Toprak yeni bir araba satın aldı.", "Kaya avukatıyla görüştü.", "Yıldız doktordan randevu aldı.",
    "Şafak bankadan kredi çekti.", "Ay ödevini bitirip uyudu.",
]
NOUN = [
    "Deniz bugün çok dalgalıydı.", "Barış için herkes çalışmalı.", "Umut her zaman vardır.",
    "Gül bahçede açtı.", "Bahar bu yıl erken geldi.", "Yağmur sabaha kadar yağdı.",
    "Derya gibi geniş bir gönlü var.", "Ümit verici gelişmeler yaşandı.", "Metin aşağıda verilmiştir.",
    "Emin olmak için tekrar kontrol et.", "Zafer kazanıldı ve bayram ilan edildi.", "Güneş batmak üzere.",
    "Kader ağlarını yine ördü.", "Su kaynadıktan sonra çayı demle.", "Nur içinde yatsın.",
    "Ada vapuru saat beşte kalkıyor.", "Ekin tarlaları sarardı.", "Onur kırıcı bir davranıştı.",
    "Özlem dolu bir mektuptu.", "Duygu yüklü bir konuşmaydı.", "Sevgi her şeyi yener.",
    "Eylül ayında okullar açılır.", "Defne yaprakları yemeğe eklenir.", "Irmak taşınca köprü yıkıldı.",
    "Pınar başında su içtik.", "Çağrı merkezi bütün gün meşguldü.", "Ufuk çizgisinde bir gemi belirdi.",
    "Doğa koruma altına alındı.", "Evren sürekli genişliyor.", "Bulut kümeleri gökyüzünü kapladı.",
    "Toprak kaymasında yol kapandı.", "Kaya parçaları yola düştü.", "Yıldız kayması gördük.",
    "Şafak sökerken yola çıktık.", "Ay bu gece dolunay.",
]


def task_a():
    print(f"A. ambiguous given name at a sentence start ({len(PERSON)} person, {len(NOUN)} word)")
    sents, y = PERSON + NOUN, [1] * len(PERSON) + [0] * len(NOUN)
    first = lambda e, s: any(x["pii_type"] in ("AD", "KISI") and x["start"] == 0 for x in e.detect(s))
    report("engine, rules only", y, [first(E0, s) for s in sents])
    report("engine, use_embeddings=True", y, [first(E1, s) for s in sents])
    ctx = [emb(s.split(" ", 1)[1]) for s in sents]
    agent = emb("koştu yürüdü düşündü anlattı sordu cevapladı okudu yazdı bekledi gülümsedi çalıştı uyudu")
    state = emb("eridi dondu kurudu büyüdü soldu esti parladı aktı titredi çatladı döküldü filizlendi")
    report("prototypes: human actions vs states", y, [cos(c, agent) >= cos(c, state) for c in ctx])
    report("logistic regression on the context, 5-fold CV", y, cv(ctx, y))


# ------------------------------------------------------------------------------------- B
HEALTH_1 = ["fibromiyalji", "Hashimoto", "lupus", "migren", "sedef", "vitiligo", "gut", "kolit", "Crohn",
            "endometriozis", "gastrit", "reflü", "guatr", "osteoporoz", "skolyoz", "glokom", "katarakt",
            "egzama", "sinüzit", "zatürre", "tüberküloz", "verem", "menenjit", "romatizma", "fıtık", "anemi",
            "talasemi", "hipotiroidi", "MS", "ALS", "obezite", "bulimia", "anoreksiya", "OKB", "DEHB",
            "disleksi", "hemofili", "sistit", "prostat", "kalp yetmezliği"]
OTHER_1 = {"hastasıyım": ["futbol", "çikolata", "kitap", "sinema", "kahve", "araba", "Galatasaray", "temizlik",
                          "müzik", "teknoloji", "dizi", "basketbol"],
           "teşhisi kondu": ["arıza", "yazılım hatası", "motor arızası", "ağ sorunu", "elektrik kaçağı",
                             "su kaçağı", "sistem açığı", "kod hatası"],
           "ilacı kullanıyoruz": ["böcek", "tarım", "fare", "yabani ot", "zirai", "pire"],
           "tedavisi gerekiyor": ["atık su", "ahşap", "deri", "yüzey", "metal"]}
HEALTH_2 = ["astigmatizma", "kist", "tendinit", "bursit", "varis", "akne", "sarılık", "kızamık", "suçiçeği",
            "kabakulak", "boğmaca", "difteri", "tetanoz", "nefrit", "pankreatit", "apandisit", "kolesterol",
            "hipotansiyon", "aritmi", "bronşektazi", "psoriyatik artrit", "çölyak", "laktoz intoleransı",
            "uyku apnesi", "vertigo", "tinnitus", "dermatit", "nezle"]
OTHER_2 = {"hastasıyım": ["voleybol", "kahve falı", "bulmaca", "satranç", "yemek programı", "rock müzik",
                          "Beşiktaş", "fotoğraf", "kamp", "bisiklet"],
           "teşhisi kondu": ["fren arızası", "kaçak akım", "çatlak boru", "donanım hatası", "kart arızası"],
           "ilacı kullanıyoruz": ["sinek", "hamamböceği", "karınca", "bitki koruma"],
           "tedavisi gerekiyor": ["parke", "mermer", "beton yüzey", "pas"]}
TEMPLATES = ["{x} tanısı konuldu.", "{x} teşhisi kondu.", "{x} tedavisi görüyorum.", "{x} hastasıyım.",
             "{x} ilacı kullanıyorum.", "{x} hastalığı yüzünden işe gelemedim."]


def task_b():
    health = emb("hastalık tanı teşhis tedavi ilaç kanser diyabet ameliyat tahlil sağlık raporu")
    everyday = emb("oyun spor film yemek tatlı alışveriş telefon makine cihaz bakım onarım tamir")
    words = {}
    for name, pos, other in [("set 1", HEALTH_1, OTHER_1), ("set 2 (held out)", HEALTH_2, OTHER_2)]:
        sents = [rng.choice(TEMPLATES).format(x=x) for x in pos] + [f"{x} {t}." for t, xs in other.items() for x in xs]
        cand = pos + [x for xs in other.values() for x in xs]
        y = [1] * len(pos) + [0] * (len(cand) - len(pos))
        words[name] = (cand, y)
        print(f"B, {name}: condition before a clinical trigger ({len(pos)} health, {len(y) - len(pos)} not)")
        found = lambda e, s, c: any(x["pii_type"] == "SAGLIK" and c.split()[-1] in x["text"] for x in e.detect(s))
        report("engine, rules only", y, [found(E0, s, c) for s, c in zip(sents, cand)])
        report("engine, use_embeddings=True", y, [found(E1, s, c) for s, c in zip(sents, cand)])
        vecs = [emb(c) for c in cand]
        report("prototype >= 0.25 (the engine's old rule)", y, [cos(v, health) >= 0.25 for v in vecs])
        report("prototypes: health vs everyday", y, [cos(v, health) > cos(v, everyday) for v in vecs])
    (c1, y1), (c2, y2) = words["set 1"], words["set 2 (held out)"]
    predict = fit(np.asarray([emb(c) for c in c1]), np.asarray(y1, dtype=float))
    report("regression trained on set 1, tested on set 2", y2, predict(np.asarray([emb(c) for c in c2])))


# ------------------------------------------------------------------------------------- C
VALUES = ["maviBalina7", "Kedi.2019", "sari-limon-42", "AyseGul!88", "Ankara06x", "deniz_yildiz9",
          "Mart2024!", "qwerty123!", "kahveCOK55", "Tr_Kod_771", "Gizem.1990", "uzay*gemisi3",
          "BeyazAt_12", "korsan77!", "Lale-2024", "yesilKapi9", "Pamuk_99", "elma.armut5"]
SECRET = ["Bunu kimseyle paylaşma: {v}", "Giriş yaparken {v} yazman yeterli.", "Hesaba {v} ile girebilirsin.",
          "Kapıyı açmak için {v} gir.", "Yeni kombinasyon {v}, eskisini unut.", "Sisteme {v} ile bağlan.",
          "Uygulamaya girerken {v} kullan.", "Kasayı {v} ile açabilirsin.", "Modeme bağlanmak için {v} lazım.",
          "Bu bilgiyi sakla, kimse görmesin: {v}"]
NEUTRAL = ["Sipariş numaranız {v}.", "Ürün kodu {v} olan cihaz stokta.", "Yeni model {v} piyasaya çıktı.",
           "Kargo takip kodu {v} olarak görünüyor.", "Faturadaki referans {v}.", "Toplantı salonu {v} katında.",
           "Etkinliğin adı {v} olarak belirlendi.", "Proje dosyası {v} adıyla kaydedildi.",
           "Takımın maskotu {v} oldu.", "Kampanya kodu {v} ile indirim kazanın."]


def task_c():
    rows = [(c, v, 1) for c in SECRET for v in rng.sample(VALUES, 4)] + \
           [(c, v, 0) for c in NEUTRAL for v in rng.sample(VALUES, 4)]
    y = [l for *_, l in rows]
    print(f"C. password-like value, no trigger word ({sum(y)} secret contexts, {len(y) - sum(y)} neutral)")
    found = lambda e, c, v: any(x["pii_type"] == "SIFRE" and v in x["text"] for x in e.detect(c.format(v=v)))
    report("engine, rules only", y, [found(E0, c, v) for c, v, _ in rows])
    report("engine, use_embeddings=True", y, [found(E1, c, v) for c, v, _ in rows])
    cred = emb("şifre parola gizli pin kodu kimlik doğrulama giriş şifresi secret key")
    ident = emb("sipariş ürün model kargo takip fatura referans salon kat etkinlik proje dosya kampanya")
    ctx = [emb(c.replace("{v}", " ")) for c, _, _ in rows]
    report("prototypes: credentials vs identifiers", y, [cos(c, cred) >= cos(c, ident) for c in ctx])
    report("logistic regression, held-out contexts", y, cv(ctx, y, groups=[c for c, _, _ in rows]))


# ------------------------------------------------------------------------------------- D
DATES = ["12/03/1985", "5 Mayıs 2024", "18.11.2016", "3 Ocak", "21/06/2023", "9 Eylül 1990", "14.02.2025"]
PRIVATE = ["Doğum tarihim {d} olarak kayıtlı.", "Kızım {d} tarihinde doğdu.", "Diş randevum {d} günü saat 10'da.",
           "Ameliyat tarihim {d} olarak belirlendi.", "Muayene için {d} tarihine gün verildi.",
           "Kargom {d} tarihinde teslim edilecek.", "Abonelik sözleşmem {d} tarihinde bitiyor.",
           "İşe giriş tarihim {d}.", "Kontrol randevum {d} sabahı.", "Eşimle {d} tarihinde evlendik.",
           "Faturamın son ödeme günü {d}.", "Kan tahlili sonuçlarım {d} tarihinde çıktı.",
           "Annemin doğum günü {d}.", "Otel rezervasyonum {d} için yapıldı.", "Mezuniyet törenim {d} günü."]
PUBLIC = ["Mahkemenin {d} tarihli kararıyla dava reddedilmiştir.", "Karar başvurucuya {d} tarihinde tebliğ edilmiştir.",
          "Kanun {d} tarihinde Resmî Gazete'de yayımlandı.", "Bireysel başvuru {d} tarihinde yapılmıştır.",
          "Yargıtay {d} tarihli kararında bozma kararı verdi.", "Seçimler {d} tarihinde yapıldı.",
          "Antlaşma {d} tarihinde imzalandı.", "Festival {d} tarihinde başlayacak.", "Erişim tarihi: {d}.",
          "Kampanya {d} tarihine kadar geçerlidir.", "Meclis {d} tarihinde toplandı.",
          "Deprem {d} günü saat 04:17'de meydana geldi.", "Bakanlar Kurulu {d} tarihinde karar aldı.",
          "Soruşturma sonucu {d} tarihinde kovuşturmaya yer olmadığına karar verildi.", "Maç {d} tarihinde oynandı."]


def task_d():
    rows = [(t, d, 1) for t in PRIVATE for d in rng.sample(DATES, 3)] + [(t, d, 0) for t in PUBLIC for d in rng.sample(DATES, 3)]
    y = [l for *_, l in rows]
    print(f"D. private vs procedural/public date ({sum(y)} private, {len(y) - sum(y)} other)")
    found = lambda e, t, d: any(x["pii_type"] in ("DOGUM_TARIHI", "OZEL_TARIH") and d.split()[0] in x["text"]
                                for x in e.detect(t.format(d=d)))
    report("engine, rules only", y, [found(E0, t, d) for t, d, _ in rows])
    report("engine, use_embeddings=True", y, [found(E1, t, d) for t, d, _ in rows])
    priv = emb("doğum günü randevu muayene ameliyat fatura kesim teslimat abonelik sözleşme mezuniyet başvuru işe giriş rezervasyon")
    pub = emb("kampanya indirim duyuru genelge resmi tatil festival konser seminer konferans basın bülteni mevzuat kanun")
    proc = emb("hüküm duruşma temyiz itiraz istinaf yönetmelik bakanlık valilik belediye savcılık iddianame tutanak savaş imparatorluk lig sezon")
    ctx = [emb(t.replace("{d}", " ")) for t, _, _ in rows]
    report("prototypes: private vs public (the old engine rule)", y,
           [(cos(c, priv) > cos(c, pub) and cos(c, priv) >= 0.15) or cos(c, priv) >= 0.35 for c in ctx])
    report("prototypes: private vs public and procedural", y,
           [cos(c, priv) > max(cos(c, pub), cos(c, proc)) for c in ctx])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for task in (task_a, task_b, task_c, task_d):
        task()
        print()
