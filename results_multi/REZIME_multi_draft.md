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
