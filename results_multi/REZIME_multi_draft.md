## 11. Više predmeta: koji je predmet i gde mu je težište (`results_multi/`)

Sistem za jednu viljušku (konačni model `mini_robust` s0) ostao je nepromenjen i ponovljiv. Dodata su tri predmeta,
a pitanje je da li se samo dodirom može prepoznati predmet i naći njegovo težište, aktivnom pretragom u više
pritisaka. Svaki korak, odluka i broj ima red u `results_multi/RUN_LOG.md` (na engleskom).

### 11.1 Predmeti

- **Registar predmeta** (`objects.json`, `objects.py`): za svaki predmet fajl i jedinice, scena, masa i težište
  izračunati iz mreže, prag CoACD-a, simetrija, da li "licem nagore / nadole" ima smisla, delovi duž dužine i stabilne
  poze. Postojeći kod za viljušku sada čita viljušku iz registra; regeneracija 40 sačuvanih pritisaka i 20 sačuvanih
  epizoda konačnog modela daju identične rezultate pre i posle te promene.
- **Novi fajlovi su u metrima** (najduža ivica 0,120 / 0,144 / 0,129 jedinica; samo metri daju dužinu pribora od
  120-250 mm). Uvezeni su kao kopije u mm, kao i viljuška.
- **Sva tri nova predmeta su ravne ploče debljine 2 mm sa istom drškom**, koja je ujedno i drška postojeće viljuške.
  Fork.stl ima tačno isti obris odozgo kao postojeća viljuška, ali je ravan, dok je postojeća viljuška savijena (5 mm
  visoka). "Kutlača" kašike je ravan ovalni deo bez udubljenja; sečivo noža širi se samo na jednu stranu.
  - Zato jedan pritisak na dršci ne može da razlikuje Fork / nož / kašiku, a težište viđeno sa drške zavisi od
    predmeta (x = -21,3 / -8,1 / -2,9 mm). To je ono što pretraga mora da razreši.
- **Prag CoACD-a po pravilu** (dva puta dorađeno pre izbora, oba puta zapisano): tanki delovi sačuvani (zupci i
  prorezi između njih, ivica sečiva, obod kašike) i ispravna sila pritiska. Druga verzija pravila izabrala je za
  viljušku prag 0,0025, ali je tada drška bila isečena drugačije nego kod noža i kašike i pritisak na nju je davao
  17 % veću silu od zadate. Takva razlika bi bila veštački trag po kome bi mreža mogla da prepozna predmet. Sa pragom
  0,01 drška je isti konveksni deo kod sva tri predmeta.
- **Provere** (`verify_stl.py`): poklapanje fizičke i vizuelne geometrije, otisak pritiska (IoU 0,999), prorezi između
  zubaca otvoreni. Provere koje ne prolaze pregledane su jedna po jedna; nijedna nije greška zavarenog predmeta koji
  sistem koristi. Sva tri predmeta su zadržana.
- **Stabilne poze** (ista pravila kao za viljušku): Fork 2 prirodne (ravno) i 4 poduprte; kašika 4 prirodne (ravno ×2,
  na ivici ×2) i 1 poduprta; nož 4 prirodne (ravno ×2, na leđima, na sečivu). Poduprte poze se ne uzorkuju.
- **Simetrija** (obris prema slici u ogledalu): viljuške i kašika simetrične (≤ 0,2 mm), nož nije (7,9 mm). Zato se
  slika u ogledalu kao augmentacija koristi samo za simetrične predmete.

### 11.2 Podaci

- Ista podešavanja kao za Mini; **po predmetu 30.000 za trening, 2.000 za validaciju, 3.000 za test** (140.000
  pritisaka, 23 min na 24 jezgra).
- Čuva se samo ono što mreža koristi: rekonstruisana dubina na ulaznoj rezoluciji (72 × 93, float16), za trening
  pritiske 5 varijanti sa strukturisanim šumom, sve oznake i seed-ovi. Ostalo se pravi ponovo iz seed-a (bit-identično).
  **Disk: 545 MB na 10.000 pritisaka za trening (sa varijantama), 110 MB za validaciju / test.**
- Provera oznaka slikom (`results_multi/data_check/`): obris iz oznaka leži tačno na signalu dubine. Pritisak u sredini
  ravne kutlače daje slab, skoro ravnomeran signal (fotometrijska rekonstrukcija vidi nagibe, ne visinu).

### 11.3 Modeli

- **A ("multi-all")**: sva četiri predmeta. **B ("multi-holdout")**: bez Fork.stl, koji u treningu nikad nije viđen.
- Isti recept kao konačni model za viljušku (mreža, trening, augmentacija, maska ivice).
- Glave: kontakt (lebdi / sto / predmet), identitet predmeta, delovi (poseban blok za svaki predmet), lice (samo gde
  je definisano) i mapa verovatnoće težišta. **Mapa je jedna, zajednička za sve predmete, ali uslovljena hipotezom o
  predmetu** (naučeni vektor predmeta dodaje se ulazu dekodera), pa pretraga dobija posebnu mapu za svaku hipotezu.
  Jedna neuslovljena mapa bi na zajedničkoj dršci pomešala težišta sva četiri predmeta.

### 11.4 Pretraga sa nepoznatim predmetom

- Zajedničko verovanje o predmetu i težištu. Svaki pritisak sa kontaktom dodaje dokaz o identitetu (glava identiteta)
  i, za svaku hipotezu, njenu mapu težišta (sa 1 % uniformne smeše, kao ranije). Verovatnoća predmeta uzima u obzir i
  koliko se pritisci slažu oko težišta pod tom hipotezom.
- Sledeći pritisak: najveći očekivani dobitak informacije o (predmet, težište): deo za težište kao ranije plus deo za
  identitet (pritisak tamo gde bi pod nekom hipotezom bila glava predmeta: zupci / sečivo / kutlača).
- Zaustavljanje: najverovatniji predmet ≥ 95 % i njegova 90 % oblast < 16 mm², ili 10 pritisaka.
- Poređenje: isti početni pritisci sa **poznatim** predmetom ("oracle"), da se vidi koliko košta nepoznat predmet.

### 11.5 Puna poza (dodatak): konvencije, glave, zajedničko verovanje, provera CAD-om

Cilj je postala puna poza predmeta na stolu: težište (x, y), yaw, id stabilne poze i koji je predmet. Težište ostaje
referentna tačka, jer je to i mesto gde robot treba da uhvati predmet.

- **Geometrija = štampana geometrija.** Vizuelne mreže imaju tačno trouglove originalnih STL fajlova; najveće odstupanje
  je 2,9·10⁻⁶ mm. Popravka, decimacija i zaglađivanje nisu bili potrebni. SHA-256 svakog originala je u `objects.json`.
  **Fork, Knife i Spoon.stl su u metrima**: pri štampi ih treba skalirati ×1000 ili štampati mm kopije iz
  `results_multi/objects/source_mm/`.
- **Štampljivost** (geometrija nije menjana): najtanje je ForkMilimeter.stl, 0,91 mm (lim na zupcima, x 15-33 mm);
  Fork.stl 1,02 mm (krajevi zubaca), Knife.stl 1,24 mm (zaobljen vrh), Spoon.stl 2,0 mm. Ništa nije tanje od 0,8 mm.
- **Konvencije** su u `objects.json`: koordinatni sistem predmeta u težištu, +x duž predmeta prema glavi; yaw je ugao
  ose +x. Poze koje se ne mogu razlikovati: Fork.stl 0/1 i Spoon.stl 0/1 (ravni i simetrični, pa je licem nagore isto
  što i licem nadole) i Spoon.stl 2/3. Provera: CAD obris postavljen samo iz označene poze leži na slici dubine
  (30 primera po predmetu).
- **Nove glave mreže:** yaw (72 polja po 5°) i stabilna poza (blok za svaki predmet). Prvi model A, bez tih glava,
  prekinut je u epohi 10 i sačuvan.
- **Zajedničko verovanje na GPU-u:** (predmet, poza) × 72 yaw-a × mreža centra 201 × 201 mm, 140 MB; po procesu
  najviše 0,5 GB. Glave mreže se množe kao nezavisne. To je pretpostavka, jer sve vide istu sliku. Pritisak proverava i
  stvarni otisak predmeta u toj pozi u odnosu na jastučić. Dobitak informacije meri se za celu pozu, ne samo za
  težište.
- **Provera CAD-om:** za 64 najverovatnije hipoteze iscrtava se očekivana dubina iz CAD-a i poredi sa
  rekonstruisanom. Na kraju se poza fino dorađuje (neprekidni x, y, yaw).
  - Prva temperatura (0,033) bila je preoštra: posle jednog pritiska na sečivo verovanje je otišlo na preslikanu pozu
    noža sa P = 1,0, i pretraga je stala pogrešno.
  - Pravilom zapisanim pre rezultata, temperatura provere izabrana je na 160 validacionih pretraga: τ = 1,0
    (94,4 % uspeha pri ADD < 5 mm, 0,6 % pogrešnih sigurnih zaustavljanja).
  - Završna dorada koristi τ = 0,027 (kalibrisano za poređenje bliskih poza uz mrežu). Ta odluka je doneta posle
    validacionih rezultata i tako je i zapisana.
- **Visina dodira** (posebna varijanta, dodata posle analize grešaka kašike): robot zna na kojoj visini je jastučić
  prvi put naišao na otpor. U simulaciji je to visina prvog dodira u odnosu na sto, uz šum od 0,3 mm (pretpostavljena
  tačnost ruke). Predviđanje iz CAD-a je najviša tačka predmeta ispod jastučića. Izmereno minus predviđeno pri tačnoj
  pozi: medijana od -0,34 do +0,09 mm. **Kućište senzora nije u predviđanju**: kad kućište udari u predmet koji stoji na
  ivici pre nego što ga gel dodirne, slika je tamna, a visina je visina kućišta (nož 41 od 973 pritiska, kašika 17 od
  1.123). U polovini preostalih grešaka noža i kašike postoji takav pritisak.
- **Neviđeni predmet sa CAD-om (C8)**: model B nema klasu za Fork.stl. Svaka stabilna poza Fork.stl je hipoteza koju
  podržava samo CAD: identitet i mapu težišta pozajmljuje od najsličnije poznate klase (fork_stl), a proverava se
  svojim CAD-om. Naknadna varijanta `--cad-align` (odlučena posle prvog rezultata, prvo proverena na validaciji) čita
  pozajmljenu mapu pomerenu za razliku težišta i ne kažnjava hipotezu kroz glavu stabilne poze.

### 11.6 Rezultati (`results_multi/pose_summary.md`, ponovo izračunato iz sačuvanih epizoda)

300 test početaka po predmetu, isti za svaku varijantu. ADD koristi iste tačke tela i kad je stabilna poza pogrešna;
ADD-S samo za parove poza koje se ne mogu razlikovati. Pogrešan predmet je neuspeh.

**Model A (sva četiri predmeta):**

| predmet | jedan pritisak: ADD < 5 mm | pretraga: predmet / težište / yaw / ADD < 5 mm / pritisaka | + visina: predmet / ADD < 5 mm / pritisaka |
|---|---|---|---|
| ForkMilimeter | 60,3 % | 100 % / 0,50 mm / 0,5° / 100 % / 2,26 | 99,7 % / 99,7 % / 2,17 |
| Fork.stl | 59,9 % | 99,7 % / 0,42 mm / 0,3° / 99,7 % / 2,27 | 100 % / 100 % / 2,11 |
| Knife.stl | 36,4 % | 99,7 % / 0,35 mm / 0,2° / 93,0 % / 3,76 | 100 % / 98,3 % / 3,24 |
| Spoon.stl | 50,2 % | 95,3 % / 0,29 mm / 0,4° / 93,7 % / 4,06 | 99,3 % / 98,3 % / 3,74 |

- Jedan pritisak nije dovoljan: na zajedničkoj dršci predmet se pogađa, a nož i kašika se okrenu za 180° u 16-17 %
  pritisaka. Glave predmeta mreža razlikuje dobro (zupci, sečivo, kutlača: 90-95 %).
- Pretraga nalazi celu pozu: težište 0,3-0,5 mm, yaw 0,2-0,5°, u 2,3-4,1 pritiska. Kalibracija je dobra (pravo težište
  u 90 % oblasti u 88-99 %, pravi yaw u 90 % intervalu u 92-98 % epizoda).
- Nepoznat predmet košta malo: u odnosu na pretragu kojoj je predmet rečen, kašika gubi 4,6 poena i treba joj 0,6
  pritiska više; ostali do 1,3 poena. Sa visinom kašika dostiže pretragu sa poznatim predmetom (98,3 %).
- Dobitak informacije za celu pozu je bitan za viljuške: pretraga samo za težište treba 3 pritiska više i gubi 24 /
  7 poena, jer ne proverava zupce, a samo po njima se dve viljuške razlikuju. Za nož i kašiku razlike nema.
- Provera CAD-om tokom pretrage skoro ništa ne menja; završna dorada spušta grešku težišta sa 0,75-0,83 na 0,29-0,50 mm
  i yaw sa 1,2-1,3° na 0,2-0,5°.
- Visina dodira: kašika +4,7 poena, nož +5,3 poena, nestaju greške "ravna kutlača, nema dodira".
- Model A prema modelu samo za viljušku, sa poznatim predmetom: medijana greške težišta +0,02 mm, p90 +0,2 mm.
- Robusnost (jedan pritisak): šum, zamućenje i drift skoro ništa ne menjaju; skala ±2 % najviše smeta nožu i kašiki.

**Model B (Fork.stl nikad viđen) na Fork.stl:**

| varijanta | predmet | težište | ADD < 5 mm | pritisaka |
|---|---|---|---|---|
| sa CAD-om, kako je planirano (C8) | **8,3 %** | **2,73 mm** | 8,3 % | 8,17 |
| sa CAD-om + visina | 92,0 % | 2,10 mm | 91,7 % | 3,92 |
| sa CAD-om, poravnato (naknadno) | 58,0 % | 0,47 mm | 58,0 % | 9,07 |
| sa CAD-om, poravnato (naknadno) + visina | 100 % | 0,26 mm | 100 % | 3,60 |
| (model A, koji je Fork.stl video) | 99,7 % | 0,42 mm | 99,7 % | 2,27 |

- **Kako je planirano, CAD ne vraća neviđenu viljušku**: B je u 92 % pretraga nazove ForkMilimeter (isti obris, ali
  savijen). Obris nađe tačno (ADD-S 0,82 mm), a težište promaši za razliku težišta dve viljuške (2,96 mm).
- Dva uzroka, nađena posle rezultata: hipoteza samo sa CAD-om gubila je ~0,6 nats po pritisku kroz glavu stabilne poze,
  a mapa težišta pokazuje težište ForkMilimeter-a.
- Visina sama popravlja identitet (ravna 2 mm prema savijenoj 5 mm), poravnanje popravlja težište; zajedno B nalazi pozu
  neviđene viljuške jednako dobro kao A, koji je na njoj treniran. Oba sastojka su naknadna.
- B na sopstvenim predmetima je kao A; sa visinom kašika 100 % / 98,3 %.
