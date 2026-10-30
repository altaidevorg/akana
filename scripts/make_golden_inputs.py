"""Writes the inputs of the golden tests; the tests fill in the expected outputs.

    python scripts/make_golden_inputs.py
    UPDATE_GOLDEN=1 cargo test -p akana-core --test golden_vectors
    AKANA_GOLD_WRITE=morphology_snapshot cargo test -p akana-core --test morphology_snapshot

- crates/akana-core/tests/data/embeddings/golden.jsonl: 160 court-decision chunks (32 per
  subset, every 20th decision of the corpus sample's eval split, chunked as an index does:
  SentenceChunker 256) and the hand-written edge cases below. Court decisions are official
  texts (FSEK art. 31) and may be reproduced.
- crates/akana-core/tests/data/morphology/top_types.txt: the 5,000 most frequent word types
  of the court eval sample and the 3,000 most frequent of RagTurk's, without counts or text.

Run it only to change the inputs; the expected outputs live in the files the tests write.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / ".cache" / "morph_sample" / "eval"
SUBSETS = ["aym_bb", "aym_norm", "danistay", "emsal", "yargitay"]

EDGE_CASES = [
    "",
    " ",
    ".",
    "a",
    "﻿Metnin başında bayt sırası işareti var.",
    "Sıfır​genişlikli boşluk ve birleştirici‍karakterler.",
    "TBMM'de kabul edilen kanun 1924'te Resmî Gazete'de yayımlandı.",
    "Av. Ahmet Yılmaz ile Tic. ve San. Ltd. Şti. arasındaki sözleşme feshedildi.",
    "İİK'nın 68. maddesi ve TCK.nun 53. maddesi uyarınca karar verildi.",
    "Dava 12.03.2019 tarihinde açılmış, 1.250.000,00 TL tazminat istenmiştir.",
    "Saat 17.00'de toplanan kurul 3'er kişilik komisyonlar oluşturdu.",
    "Bilirkişi raporunda kusur oranları %75 ve %25 olarak belirlenmiştir.",
    "KANUN HÜKMÜNDE KARARNAME İLE DÜZENLENEN HUSUSLAR",
    "CIA'nın ve FBI'ın raporları KITAPLAR arasında bulundu.",
    "Prof. Dr. Ayşe Kaya, XIX. yüzyıl edebiyatı üzerine bir konferans verdi.",
    "Türkiyede ve istanbulda yaşayanların sayısı arttı.",
    "Kur'an-ı Kerim'in meali yayımlandı.",
    "ı i I İ ç Ç ğ Ğ ö Ö ş Ş ü Ü â î û",
    "e-posta adresine gönderilen belge @akana ve #nlp etiketleriyle paylaşıldı.",
    "Napıyorsun ya, bi dakka bekle!",
    "Gelmeyecekmişsiniz gibi davranmayın, hepinizi bekliyoruz.",
    "Avrupalılaştıramadıklarımızdanmışsınızcasına konuşuyorsunuz.",
    "Çoğunluk görüşüne katılmıyoruz.",
    "Bana bir kitap ver.",
    "Sözümü anla!",
    "Siyasi partiler ister iktidarda ister muhalefette olsunlar, kanuna uymak zorundadır.",
    "Kahverengi ayakkabılarını giyip bilgisayarın başına oturdu.",
    "Cumhurbaşkanlığı Kararnamesi ile zeytinyağı ihracatına yeni düzenleme getirildi.",
    "Denizaltı, Karadeniz'in derin sularında üç gün boyunca seyretti.",
    "Anayasa Mahkemesi başvuruyu oybirliğiyle kabul edilemez buldu.",
    "Hâlâ gelmediler, hala bekliyoruz.",
    "Davacının hakları korunmalıdır.",
    "Yapılacak bir şey kalmadı.",
    "1, 2, 3, 4, 5, 10, 100, 1000, 1.000.000, 3,14, IV, ii, XL'i",
    "Ve, veya, ile, ama, fakat, ancak, çünkü, eğer, ise.",
    "ABD'nin ve AB'nin yaptırım kararları uluslararası ticareti etkiledi.",
    "COVID-19 salgını sırasında uzaktan eğitime geçildi.",
    "Ekmek aldım. Kanun, herkes için eşittir. Umut her zaman var.",
    "ölüm ileri devam ay tabi teslim oy geri",
    "Mahkemenin kararı kesinleşti; dosya Yargıtay'a gönderildi.",
]


def court_chunks() -> list[str]:
    import akana

    chunker = akana.SentenceChunker(chunk_size=256)
    texts = []
    for subset in SUBSETS:
        decisions = (SAMPLE / f"{subset}.txt").read_text(encoding="utf-8").split("\n\x1e\n")
        picked = []
        for decision in decisions[::20]:
            chunks = chunker.chunk(decision)
            if chunks:
                picked.append(chunks[0].text)
            if len(picked) == 32:
                break
        texts.extend(picked)
    return texts


def top_types() -> list[str]:
    def top(path: Path, n: int) -> list[str]:
        counts = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            w, _, c = line.partition("\t")
            if w and c:
                counts[w] += int(c)
        return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    court = Counter()
    for subset in SUBSETS:
        for line in (SAMPLE / f"{subset}.types.tsv").read_text(encoding="utf-8").splitlines():
            w, _, c = line.partition("\t")
            if w and c:
                court[w] += int(c)
    words = [w for w, _ in sorted(court.items(), key=lambda kv: (-kv[1], kv[0]))[:5000]]
    seen = set(words)
    for w in top(SAMPLE / "ragturk.types.tsv", 3000):
        if w not in seen:
            words.append(w)
            seen.add(w)
    return words


def main() -> None:
    golden = ROOT / "crates/akana-core/tests/data/embeddings/golden.jsonl"
    golden.parent.mkdir(parents=True, exist_ok=True)
    with golden.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"sparse_spec": "", "fingerprint": ""}) + "\n")
        for text in court_chunks() + EDGE_CASES:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    types = ROOT / "crates/akana-core/tests/data/morphology/top_types.txt"
    types.write_text("\n".join(top_types()) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {golden} and {types}")


if __name__ == "__main__":
    main()
