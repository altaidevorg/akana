//! Sample texts for the detector's and the humanizer's tests, taken from the detector's
//! evaluation data (not written by hand): WildChat answers (allenai/WildChat, ODC-BY) and
//! Turkish Wikisource texts (public domain).

/// An OpenAI answer (WildChat, o1-preview, id 151b38df05e571c8).
pub(in crate::style) const AI_TEXT: &str = r#"En katma değerli ve ölçeklenebilir kompozit malzemeler arasında **karbon fiber takviyeli polimerler (CFRP)** öne çıkmaktadır. CFRP, yüksek mukavemet/ağırlık oranı, mükemmel yorulma direnci ve korozyon direnci gibi üstün mekanik özellikleri sayesinde havacılık ve uzay sanayii, otomotiv, rüzgar enerjisi ve spor ekipmanları gibi çeşitli endüstrilerde yaygın olarak kullanılmaktadır.

Bu malzemelerin üretimi, reçine aktarım kalıplama (RTM) ve otoklav laminasyonu gibi ölçeklenebilir üretim teknikleriyle gerçekleştirilebilir. Ayrıca, otomasyon ve robotik teknolojilerinin entegrasyonu ile üretim süreçleri daha verimli hale getirilebilir, bu da büyük ölçekli üretime olanak tanır.

Karbon fiber kompozitlerin yüksek katma değeri, ürünlerin performansını artırırken ağırlığı azaltması ve enerji verimliliğine katkıda bulunmasından kaynaklanmaktadır. Bu özellikler, özellikle yakıt tasarrufu ve azaltılmış emisyonların önemli olduğu sektörlerde büyük ekonomik ve çevresel faydalar sağlamaktadır."#;

/// A folk poem (Turkish Wikisource, id 984244ffc4199c8c).
pub(in crate::style) const POEM: &str = r#"İki keklik bir kayada ötüyor
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

/// An OpenAI chat answer the academic rule flags (WildChat, gpt-4o, id d3d3a89f0be18699).
pub(in crate::style) const ACADEMIC_CUES_TEXT: &str = r#"Elbette, Chain-of-Thought Prompting yöntemiyle adım adım inceleyelim:

1. **"Narrator" kelimesi**: 
   - "Narrator" kelimesindeki harflere bakalım: N, a, r, r, a, t, o, r.
   - Burada üç tane 'r' harfi var.

2. **"Strawberry" kelimesi**:
   - "Strawberry" kelimesindeki harflere bakalım: S, t, r, a, w, b, e, r, r, y.
   - Burada da üç tane 'r' harfi var.

3. **"Kelimeler arasındaki toplam 'r' sayısı"**: 
   - "Narrator" kelimesinde 3 'r' + "Strawberry" kelimesinde 3 'r' = Toplam 6 'r' harfi bulunuyor.

Sonuç olarak, "Narrator" ve "Strawberry" kelimelerindeki toplam 'r' harfi sayısı 6."#;

/// A Nasreddin Hoca story (Turkish Wikisource, id 176742819e0e7a3f).
pub(in crate::style) const HUMAN_TEXT: &str = r#"Nasreddin Hoca'nın bir oğlu varmış. Ne denirse tam tersini yaparmış. Bir gün Nasreddin Hoca ile oğlu Akşehir'e gidip oradan iki çuval tuz almışlar. Eşeğe yükleyip köylerine dönmek için yola koyulmuşlar. Köylerinin yakınındaki dereye geldiklerinde, Nasreddin Hoca, derenin en sığ yerinden karşı tarafa geçmiş. Oğlu da eşeği sudan geçirmeye çalışıyormuş. Ama eşeğin sırtındaki çuvallardan biri, neredeyse suya değmek üzereymiş. Hoca, oğlunun ters huyunu bildiği için seslenmiş:
– Sevgili oğlum. Çuvallardan biri suya değecek. Biraz daha asıl da, iyice suya gömülsün.
Ne var ki çocuğun o gün uysallığı üzerindeymiş. Tutup, babasının dediğini yapmış. Neticede iki çuval birden suyun içine düşmüş. Nasreddin Hoca:
– Ne yapıyorsun şaşkın oğlum! diye bağırmış, ama bu arada tuzlar da suda eriyip gitmiş.
Nasreddin Hoca'nın oğlu, babasının kızgınlığı karşısında şöyle söyleniyormuş:
– Kırk yılda bir baba sözü dinleyelim dedik, gene de yaranamadık."#;
