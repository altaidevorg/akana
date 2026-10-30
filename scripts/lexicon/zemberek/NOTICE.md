# Zemberek dictionaries

The `*.dict` files and `turkish-letter-names.txt` here are the default dictionaries of
[zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp)
(`morphology/src/main/resources/tr`, `TurkishDictionaryLoader.DEFAULT_DICTIONARY_RESOURCES`)
by Ahmet A. Akın and contributors, licensed under the Apache License 2.0. They were taken
unmodified from the copy shipped in Zeyrek 0.1.3 (`zeyrek/resources/tr`, 2022-12-29).

`scripts/build_full_lexicon.py` reads them to build
`crates/akana-core/src/morphology/zemberek_lexicon.txt`. The akana lexicon was first built
from zemberek-nlp commit 037202c7e18b977c95181a76db05cb1e95846554; with the old build script,
these copies produce the same lexicon except for 4 lines (ağlaşmak missing; içişleri,
yapıştırıcı and mö added).

`scripts/fetch_zemberek_dicts.py` checks the files against the SHA-256 below (the lexicon
build runs the check first) and with `--fetch` re-creates them from the pinned wheel,
`zeyrek-0.1.3-py2.py3-none-any.whl` (SHA-256
23649bb49322a52d1e94959029b047fa4037bc540762819feb1096aa976b25b5).

SHA-256:

| File | SHA-256 |
|---|---|
| abbreviations.dict | 1f7ad0ffdbb4a833850bb53efb131c7e6fa89e65905575f90b38855462ca97e4 |
| master-dictionary.dict | f2613501adbe6cf258d20b399704a5ec55ec2b61ac42f082c990fae3745471f1 |
| non-tdk.dict | b2eaebed3d0583db7c7e74ef38d936d56a3a0b735fedc901da6520f7b2f1cd45 |
| person-names.dict | 0f4bc4d607158266e0098a2b6c57eed661756056391a358f27bc4e2345b2fb14 |
| proper-from-corpus.dict | af65f1c5a092bf49dc4773f8b614ee859b033c0d101de651a9e08b5e06d92005 |
| proper.dict | 5704f4017854c474442cce246443b9fec5967799a968ab9d7eb0cce930e32178 |
| turkish-letter-names.txt | 6658683d65107e4a32c0d8ceb485f9db30646a1284fcaf11ca3cc894d9f1565c |
