# Turkish PII annotation guideline

You mark personal data in short Turkish texts (court decisions, Wikipedia, news, tweets, reviews,
voice-assistant requests, prompts written to chatbots, comments, maths word problems). Mark what
is there, whatever the policy later does with it: public figures, call-centre numbers and public
dates are marked too, with an attribute that says so.

## How to mark

- Copy each value **exactly** as written: same letters, case, spacing and punctuation inside the
  value. Never correct spelling or add characters.
- A case suffix after an apostrophe is **not** part of the value: `Ahmet Yılmaz'ın` → `Ahmet
  Yılmaz`; `05321234567'den` → `05321234567`. A suffix glued on without an apostrophe stays:
  `ahmetin` → `ahmetin`.
- Titles and roles are not part of a name: `Sayın Ayşe Kaya`, `Av. Mehmet Demir`, `Ali Bey`,
  `Dr. Selin Ak` → `Ayşe Kaya`, `Mehmet Demir`, `Ali`, `Selin Ak`.
- Leave out surrounding punctuation and quotes. Keep punctuation that belongs to the value
  (`A. Yılmaz`, `No:12 D:3`, `+90 (532) 111 22 33`).
- A value inside another value is not marked twice: the name inside `ahmet.yilmaz@gmail.com` is
  part of the e-mail.
- If the same string occurs several times with the same type, list it once. If it occurs with
  different types, give `occurrence` (1 = first occurrence in the text).
- Anonymised placeholders (`...`, `A.B.`, `X`, `***`) are not values; `M.K.` initials of a real
  person are a PERSON.

## Types

**PERSON**: a person's name: full name, given name or surname alone, nickname used as a name,
initials with a surname (`A. Yılmaz`, `Figen L. Başar`), in any case (`AHMET YILMAZ`, `ahmet
yılmaz`) or spelling (`ahmet yilmaz`).
- `public: true` for public figures, historical people, authors, athletes, politicians, fictional
  or mythological characters; `false` for private individuals or when unsure.
- Not a PERSON: a name inside the name of an organisation, place, street, building, product or
  event (`Ahmet Yesevi Üniversitesi`, `Atatürk Caddesi`, `Fatih Sultan Mehmet Köprüsü`,
  `Mustafakemalpaşa`, `Nobel Ödülü`); a word that is a name but used as a common word (`deniz
  kenarı`, `umut dolu`, `barış görüşmeleri`, `gül fidanı`).

**USERNAME**: social-media handles and account or login names (`@ayse_k`, `kullanıcı adı:
deniz1985`).

**ADDRESS**: a postal address or location precise enough to find a home or workplace: street,
neighbourhood (mahalle) or site with a building, door or flat; include number, block, floor, flat,
district and province when they run on (`Atatürk Mah. Gül Sk. No:12 D:3 Kadıköy/İstanbul`). A
province, district, city, country or region alone is **not** an address (`Ankara'da yaşıyorum`).

**PHONE**: any phone or fax number. `public: true` for corporate, call-centre, emergency and public
service numbers (444 xx xx, 0850, 0800, 112, a company's switchboard).

**EMAIL**: any e-mail address. `public: true` for role or corporate mailboxes (`info@`, `destek@`,
`kvkk@`, `noreply@`).

**URL**: any web address. `personal: true` when it identifies a person or carries a private token
or record id (a person's profile, a link with `?token=`, `/dosya/8GGRJMM/`, a personal file share);
`false` for ordinary public pages, also with ordinary query parameters (`?sayfa=2`, `?lang=tr`,
`?kategori=duyurular`) and in logs and configuration (a help-page constant, `host=`, `path=`).

**ID**: an identifier of a person, an account or a record, with `subtype`:
- `national`: TCKN, VKN, passport, driving licence, SGK number;
- `financial`: IBAN, account number, card number or part of it (`son dört hanesi 2395`);
- `customer`: customer, member, subscriber, policy, contract or employee number;
- `transaction`: order, ticket, invoice, cargo tracking, application or reference number;
- `vehicle`: plate, VIN, engine number;
- `device`: IMEI, MAC, IP address, device id;
- `case`: the number of a court or administrative case or file (`2019/123 E.`, `E:2015/26`,
  `Başvuru Numarası: 2014/8875`);
- `other`.
Not an ID: amounts, prices, quantities, percentages, years, law and article numbers (`6100 sayılı
Kanun`, `12. madde`), product or model codes, version strings, page numbers, and the code or number
of a public document, notice, circular, regulation, standard, publication or web page (the code
printed on a public bulletin, `genelge no 2024/7`, `duyuru no`, the document id of a public page,
`Resmî Gazete sayı 30495`). A document number is an ID only when the document is about a
particular person or their business (`mezuniyet belgesi no`, `çalışma izni belgesi no`, a
customer's `policy_no`, a court file).

**SECRET**: passwords, PINs, one-time and verification codes, CVV, API keys, access tokens, private
keys. Not a secret: document, reference and tracking codes (an ID when personal, nothing when
public), hashes and checksums of public files, encoding names.

**DATE**: a calendar date or date-time (`12.04.2026`, `12 Nisan 2026`, `Nisan 2026`, `2027-04-14T07:01`,
`12/04 14:30`, `5 Mayıs`). Include an attached time. Do not mark years alone, weekdays or relative
dates (`yarın`, `geçen salı`, `cuma günü`).
- `personal: true` when it is about a specific individual's life or business: birth, death, marriage,
  appointment, purchase, delivery, payment, hospital stay, a private person's hearing or
  application.
- `personal: false` for public, historical or generic dates: holidays, campaigns, laws, public
  events, publication, release and update dates (`yayın tarihi`, `son güncelleme`, the
  date field of a public page or log entry), a public figure's well-known dates.

**SENSITIVE**: a special category of personal data (KVKK article 6) **stated about an identifiable
person**, including the writer (`diyabetim var`, `annem kanser tedavisi görüyor`, `Ahmet
sendikaya üye`), with `subtype`: `health`, `disability`, `religion`, `ethnicity`, `political`,
`union`, `sexual`, `criminal` (convictions, records), `biometric`, `genetic`. Mark the shortest
phrase that states it (`diyabet`, `kanser tedavisi`). A topic without a person (`kanser
araştırmaları`, `İslam sanatı`, `sendika yasası`) is **not** marked.

## Do not mark

Organisations, institutions, courts, companies; places that are not addresses; nationalities and
languages; job titles and roles; product names; amounts; ages (`35 yaşında`).
