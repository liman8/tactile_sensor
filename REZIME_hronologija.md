# Rezime: lokalizacija viljuške taktilnim senzorom (simulacija u MuJoCo)

**Cilj.** Na osnovu jednog ili više otisaka taktilnog senzora GelSight treba prepoznati viljušku i
odrediti gde je njeno težište u odnosu na senzor, odnosno kuda pomeriti senzor.

---

## 1. Simulacija senzora
- Senzor nalik GelSight-u napravljen je u MuJoCo. Lanac je: fizika kontakta → dubina utiska u
  gelu → fotometrijska RGB slika (LED osvetljenje). Slika ima 320×240 px, 0,1 mm/px.
- Postoje dva režima:
  - originalni: senzor miruje, a predmet se utiskuje u njega;
  - novi, podrazumevani: predmet leži na stolu, a senzor na „ruci“ pritiska odozgo.
- Masa ruke je podešena tako da gel ima istu mekoću kao u originalu. Za istu silu, utisak se
  razlikuje najviše ±5 %.

## 2. CAD viljuška (`ForkMilimeter.stl`)
- Uvezena je kao objekat `fork_stl`: vizuelni mesh za prikaz i 17 konveksnih delova (CoACD) za
  sudare.
- Provere:
  - kolizioni oblik odstupa od CAD površine najviše 22 µm;
  - otisak se poklapa sa renderom (IoU 0,9996);
  - sila na senzoru jednaka je zadatoj (medijan odstupanja 0,1 %).
- Prvobitnih 15 objekata daje bit-identične rezultate kao pre dodavanja viljuške, dakle ništa
  nije pokvareno.

## 3. Klasični detektor „viljuška / nije viljuška“
- Metod: fotometrijski stereo → mapa visine → ručno birana obeležja → RandomForest.
- Radi i u novom režimu bez ponovnog treniranja (tačnost 98,3 %).
- Posle dodavanja `fork_stl` u trening, CAD viljušku prepoznaje u 99,3 % slučajeva. Cena je
  više lažnih „viljuški“ na češlju (2,7 %).

## 4. Skup podataka (`collect_dataset.py`)
- Viljuška se postavlja u stabilne položaje: licem nadole ili nagore, sa nasumičnim mestom i
  rotacijom.
- Pritisci su nasumični: deo viljuške, sila 0,2–10 N, nagib senzora do 3°.
- Svaki uzorak se iz svog seed-a regeneriše bit-identično.
- Obim:
  - `dataset/`: 10.000 uzoraka;
  - `dataset_v3/`: još 40.000 za trening i 5.000 za test (28 GB);
  - `dataset_mini/`: 106.000 uzoraka sa senzorom veličine Mini-ja (19 GB).
- Provere:
  - oznake se poklapaju sa slikama (IoU 0,993);
  - zbir kontaktnih sila jednak je očitavanju senzora sile (razlika najviše 3,6·10⁻¹⁵ N);
  - nema preklapanja između trening i test skupa.

## 5. Neuronska mreža za težište (v1, `train_centre.py`)
- **Ulaz** je dubina rekonstruisana iz RGB slike, kao kod pravog senzora, a ne tačna dubina iz
  simulatora.
- **Izlazi:**
  - (a) kontakt: nema / sto / viljuška;
  - (b) deo viljuške: zupci, dlan, vrat, drška;
  - (c) strana: licem nagore ili nadole;
  - (d) mapa verovatnoće položaja težišta (150×150 ćelija od 2 mm), iz koje se dobija
    „90 % oblast“.
- Mreža je ResNet sa 11,2 M parametara.
- **Rezultat:** medijan greške 1,0 mm, p90 2,1 mm, težište je u 90 % oblasti u 97 % slučajeva.
- **Važno zapažanje:** na dršci mreža koristi to što se drška sužava (0,05 mm po mm dužine). U
  simulaciji to radi, ali na pravom senzoru i drugoj viljušci verovatno ne bi.
- **Slabosti:** osetljiva je na šum (glava za kontakt šum vidi kao viljušku) i na grešku razmere
  mm/px.

## 6. v2: robusnost, pretraga sa više pritisaka, kalibracija (`centre_v2.py`)
- **Pretpostavka:** viljuška je pričvršćena za sto, pa se ne pomera pri pritisku.
- **Trening sa augmentacijom:** šum, zamućenje, drift osvetljenja, tekstura gela, greška razmere
  ±3 %.
- **Rezultat:**
  - mreža je neosetljiva na šum do 0,1 mm, zamućenje do 0,4 mm i drift osvetljenja do 10 %;
  - na podignutim zupcima greška je pala sa 97 mm na 1,7 mm.
- **Pretraga sa više pritisaka.** Mape iz više pritisaka se množe (Bajesovo kombinovanje), a
  sledeći pritisak se bira prema tome. Rezultati:
  - 90 % oblast pada ispod 16 mm² (otprilike kvadrat 4×4 mm) u 97–100 % epizoda, za medijan od
    2 pritiska;
  - nijedna epizoda ne završi sa greškom većom od 5 mm.
- **Preciznost (pričvršćena viljuška, simulacija, model na 40k):**
  - jedan pritisak: medijan greške 0,93 mm, p90 2,1 mm;
  - kad oblast prvi put padne ispod 16 mm² (2 pritiska u 86–92 % epizoda): medijan 0,73 mm,
    p90 1,4 mm;
  - na kraju pretrage (do 10 pritisaka): medijan 0,6–0,65 mm, p90 1,1–1,3 mm.
- **Zahtev za kalibraciju:** razmera mm/px mora biti poznata na ±2 %. Dve simulirane procedure
  to postižu:
  - etalon sa dve kuglice: u 95 % ponavljanja greška ≤ 0,09 % (najgore 0,12 %);
  - poznati pomeraji robota: u 95 % ponavljanja ≤ 0,31 % (najgore 0,42 %).

## 7. v3: trenutni zadatak (`centre_v3.py`, `abs_baseline.py`)

### 7.1 Da li više podataka pomaže?
Ista mreža je trenirana na 2k, 7,2k, 20k i 40k pritisaka i testirana na novih 5.000.

| trening | medijan greške | p90 greške | tačnost dela | tačnost strane |
|---|---|---|---|---|
| 2k | 1,81 mm | 9,55 mm | 82,1 % | 78,9 % |
| 7,2k | 1,11 mm | 3,34 mm | 89,4 % | 84,5 % |
| 20k | 0,97 mm | 2,31 mm | 91,8 % | 87,0 % |
| **40k** | **0,93 mm** | **2,06 mm** | **92,8 %** | **88,9 %** |

- **Odgovor:** da, ali sve manje. Prelaz 7,2k → 20k smanjio je p90 za 1,0 mm, a 20k → 40k samo za
  0,25 mm. Intervali poverenja su dobijeni bootstrap metodom, sa uparenim razlikama.
- Veći dobitak je u klasifikaciji:
  - prepoznavanje vrata: +6 procentnih poena;
  - strana drške: +4 procentna poena.
- **Ograda:** svi modeli su trenirani 120 epoha, pa veći skup ima i više koraka optimizacije.
  Udeo „više podataka“ i udeo „više koraka“ nisu razdvojeni.

### 7.2 Ponovljivost (tri seed-a)
- Model na 40k je treniran 3 puta, istim receptom i sa različitom slučajnom inicijalizacijom.
- Rasipanje između tri treniranja je vrlo malo:

  | | srednja vrednost ± std |
  |---|---|
  | medijan greške | 0,927 ± 0,003 mm |
  | p90 greške | 2,069 ± 0,005 mm |
  | tačnost dela / strane | 93,0 ± 0,2 % / 89,0 ± 0,1 % |
  | pretraga: < 16 mm² za najviše 2 pritiska | 88,6 ± 1,0 % |
  | pretraga: greška > 5 mm | 0 % u svakom seedu |

- Zato su dobici od 20k → 40k stvarni, a ne slučajni. Smanjenje p90 za 0,25 mm iznosi oko 50
  standardnih devijacija, a +6 poena na vratu oko 8.

### 7.3 Poređenje sa metodom bez treniranja (CAD baseline)
- **Metod:** rekonstruisana dubina poredi se sa šablonima iz CAD modela, za svaki položaj i
  rotaciju. Prvo ide gruba pretraga (normalizovana korelacija), zatim fino podešavanje.
  Fitovana su samo 4 parametra.

| | mreža (40k) | CAD baseline |
|---|---|---|
| medijan greške, jedan pritisak | **0,94 mm** | 8,7 mm |
| najverovatnija ćelija unutar 5 mm | **97 %** | 41 % |
| pretraga: uspeh (< 16 mm²) | **100 %** | 48 % |
| pretraga: završi > 5 mm od cilja | **0 %** | 32 % |
| vreme po pritisku, GPU / CPU (1 nit) | **1,6 ms / 69 ms** | 88 ms / 3,05 s |

- **Zašto je CAD slab:**
  - gruba faza gubi tačan položaj; kod zubaca je tačan položaj među kandidatima u samo 39 %
    slučajeva;
  - vrat i drška su ravne trake, pa im pomereni položaji odgovaraju gotovo jednako dobro.
- **Prednosti CAD-a:** ne treba mu trening (šabloni se prave za 35 s).
- **Cena mreže:** 40.000 simuliranih pritisaka i oko 3,5 h treninga na GPU.

### 7.4 Skripte za pravi senzor (GelSight Mini)
Napisane su tri skripte:
- `record_real.py`: snimanje;
- `calibrate_real.py`: mm/px iz etalona ili iz pomeraja robota;
- `measure_noise.py`: merenje šuma i drifta, u istim jedinicama kao augmentacija.

Analitički deo je proveren na simuliranim snimcima sa poznatim tačnim vrednostima. Greška
kalibracije je 0,04–0,19 %.

### 7.5 Senzor veličine GelSight Mini (`results_centre/mini/`)
- **Simulator** sada ima i senzor veličine Mini-ja: 18,6 × 14,4 mm (186 × 144 px, 0,1 mm/px).
  - Masa senzora je ista kao kod standardnog, pa gel na istu silu reaguje isto.
  - Mini slika je tačan isečak sredine standardne slike, a stari podaci se i dalje regenerišu
    bit-identično.
- **Podaci:** 100.000 trening pritisaka i 5.000 test pritisaka. Test skup ima iste seedove kao
  standardni (isti položaji viljuške i isti pritisci); razlikuje se samo senzor.
- **Modeli:** isti recept, trenirani na 20k, 40k i 100k pritisaka (100k: 3 h treninga).

**Usput pronađena greška u mom postupku.** U treningu se svaka ulazna slika prevzorkuje (rotacija i
skaliranje 0,97–1,03, nikad tačno 1). Na testu se slika ne prevzorkuje, pa je mreža nikad nije
videla takvu.
- Svih šest modela radi bolje kad se ulaz smanji za 1 % (razmera 0,99).
- Vrednost 0,99 izabrana je na validacionom skupu.
- Uzrok nije razjašnjen: nije zamućenje i nisu ivični pikseli.
- Ovo važi i za sve ranije brojeve. Za standardni model na 40k:
  - p90 greške: 2,07 → 1,91 mm;
  - konačna greška pretrage: 0,65 → 0,56 mm.
- Brojevi ispod su sa razmerom 0,99.

**Jedan pritisak:**

| | medijan | p90 | p90 na dršci | tačnost strane |
|---|---|---|---|---|
| standardni, 40k (3 seeda) | 0,91 mm | 1,91 mm | 2,4 mm | 89,0 % |
| Mini, 40k | 0,91 mm | 2,18 mm | 13,7 mm | 85,2 % |
| Mini, 100k | **0,87 mm** | **1,73 mm** | 13,1 mm | 86,5 % |

- **Pri istih 40k pritisaka** manje polje košta malo: medijan je isti, p90 je gori za 0,27 mm, a
  tačnost strane za 3,8 poena.
- **Skoro sva cena je na dršci.** Mini vidi samo 18,6 mm dugačke, ravne trake. U 13,5 % pritisaka na
  dršci zato pogreši mesto duž drške za više od 5 mm; kod standardnog senzora to je 5,6 %.
  - Mreža tada sama „zna“ da nije sigurna: 90 % oblast je velika (medijan 216 mm²) i sadrži težište
    u 88 % takvih slučajeva.
  - 100k podataka to ne popravlja.
- **100k podataka se isplati.** Mini na 100k je ukupno bolji od standardnog senzora na 40k, osim
  na dršci i u tačnosti strane.

**Pretraga sa više pritisaka (300 epizoda):**

| | konačna greška (medijan / p90) | < 16 mm² za najviše 2 pritiska | greška > 5 mm |
|---|---|---|---|
| standardni, 40k (3 seeda) | 0,56 / 1,1 mm | 91,6 % | 0 % |
| Mini, 40k | 0,59 / 1,1 mm | 90 % | 0 % |
| Mini, 100k | **0,54 / 1,0 mm** | **96 %** | 0 % |

- Sa više pritisaka Mini je jednako precizan kao standardni senzor: sledeći pritisci razrešavaju
  problem drške.
- **Bez razmere 0,99** Mini pretraga je lošija: 1,07 mm, a 90 % oblast sadrži težište samo u 50 %
  epizoda. Uzrok je pronađen ponavljanjem epizoda pritisak po pritisak:
  - mape kasnijih pritisaka imaju veću grešku;
  - pretraga ih množi kao da su greške nezavisne, a nisu.

**Ispravka u treningu (`results_centre/mini_exact/`).** Mini 100k je treniran ponovo, tako da četvrtina
trening slika ne bude prevzorkovana: te slike dobijaju samo okret za 0° ili 180° i/ili ogledalo.
- Sve ostalo je isto: isti podaci i isti broj epoha.
- Trenirana su tri seeda, svaki po 3 h.

| Mini 100k | ulaz | medijan | p90 | konačna greška pretrage | težište u 90 % oblasti na kraju pretrage |
|---|---|---|---|---|---|
| stari recept | kako je snimljen | 0,92 mm | 2,23 mm | 1,07 mm | 50 % |
| stari recept | razmera 0,99 | 0,87 mm | 1,73 mm | 0,54 mm | 89 % |
| **ispravljen (3 seeda)** | **kako je snimljen** | **0,88 mm** | **1,84 ± 0,07 mm** | **0,71 ± 0,04 mm** | **83 %** |
| ispravljen (3 seeda) | razmera 0,99 | 0,87 mm | 1,78 ± 0,05 mm | 0,56 ± 0,03 mm | 92 % |

- **Pad na razmeri 1 je skoro nestao.** Razlika p90 između razmere 1 i 0,99 je sada 0–0,1 mm, a
  ranije je bila 0,5 mm. Time je potvrđen uzrok: mreža nikad nije videla neprevzorkovanu sliku.
  - Šta tačno mreža „uči“ iz prevzorkovanja i dalje nije poznato.
- **Za jedan pritisak razmera 0,99 više nije potrebna.**
- **U pretrazi razmera 0,99 i dalje pomaže,** za oko 0,15 mm, na sva tri seeda.
  - Ponavljanje epizoda pokazuje uzrok: pritisci iste epizode, uglavnom na dršci, dele zajedničku
    grešku od 0,78 mm, a sa razmerom 0,99 od 0,57 mm.
  - Množenje mapa takvu zajedničku grešku ne može da poništi usrednjavanjem.
  - Standardni model na 40k ima istu zajedničku grešku (0,76 mm).
- **Rasipanje između seedova je kod Mini-ja veće:** std p90 je 0,07 mm, a kod standardnog
  senzora 0,005 mm.
- **Novi nedostatak:** ispravljen model je lošiji kad je slika uvećana za 2 % (p90 oko 2,45 mm).
  Smanjenje za 2 % ne smeta. Na pravom senzoru je zato kalibracija mm/px važnija nego ranije.

**Dijagnostika: zašto pomaže razmera 0,99?**
- **Razmera 0,99 menja tri stvari:**
  - samu razmeru;
  - interpolaciju piksela (u sredini je nema, a raste ka ivicama);
  - spoljni red piksela, koji se meša sa nulama van slike i potamni na oko pola vrednosti.
- **Nije razmera.** Prevzorkovanje bez promene razmere („0,99 pa nazad“, „pola piksela pa nazad“) daje u
  pretrazi isto što i 0,99, kod sva tri modela.
- **Spoljni red piksela je jednako važan kao unutrašnjost.**
  - 0,99 bez potamnelog ruba gora je od neobrađene slike (pretraga: 1,16 umesto 0,71 mm).
  - Samo potamneli rub takođe ne pomaže.
- **Simetrija augmentacije je izmerena i u redu je:** uvećanje u treningu je ravnomerno na 0,97–1,03,
  pola slika je uvećano, a pola umanjeno.
- **Asimetrija na 1,02 nije razjašnjena.** Uvećanje škodi više od umanjenja kod svih modela, i to i
  unutar opsega treninga.
  - Mreža ne skalira pogrešno odgovor: nagib procene prema stvarnosti je u granicama 0,5 % od 1.
  - Rub koji ne odgovara razmeri je katastrofalan: 1,02 sa potamnelim rubom daje p90 oko 5 mm.

**Pravac greške u pretrazi (u sistemu viljuške).**

| ulaz | pomak duž viljuške |
|---|---|
| kako je snimljen | −0,5 mm (stari Mini −1,0 mm) |
| razmera 0,99 | −0,2 mm |
| razmera 1,02 (ispravljen model) | +0,66 mm |

- Pomak je uvek duž ose viljuške. Znak minus znači ka kraju drške, a plus ka zupcima. Bočno je skoro nula.
- **Kad se taj stalni pomak oduzme,** konačna greška je oko 0,55 mm za svaki ulaz i svaki model. Cela
  razlika između obrada ulaza je, dakle, taj pomak.
- **Gde nastaje:** pretraga se završava kod težišta, na dršci.
  - Drška se sužava za samo 0,05 mm po milimetru, pa 1/8 piksela prividne širine znači 0,5 mm duž drške.
  - To je moje tumačenje i nije provereno.
- **Posledica za pravi senzor:** način na koji se slika prevzorkuje i izgled njenog ruba pomeraju odgovor
  duž drške za do oko 1 mm.

## 8. Šta nije provereno
- **Sve je simulacija.** Pravi senzor i robot nisu korišćeni; snimanje sa hardvera nije
  pokrenuto.
- **GelSight Mini:**
  - polje je uzeto iz specifikacije;
  - debljina (4,25 mm) i mekoća pravog gela nisu modelovane ni izmerene;
  - Mini modeli po starom receptu trenirani su samo jednom (jedan seed); ispravljen 100k ima tri.
- **Prevzorkovanje ulaza:**
  - ponovni trening je urađen samo za Mini 100k i samo sa jednim udelom (25 %) neprevzorkovanih
    slika; standardni modeli i Mini 20k/40k i dalje imaju ovaj problem;
  - zašto je ispravljen model lošiji na uvećanoj slici nije razjašnjeno; pretraga sa uvećanjem 1,02
    pokrenuta je samo za Mini modele;
  - pomak duž drške izmeren je samo tamo gde se pretraga završava (kod težišta). Za pojedinačne
    pritiske na drugim delovima viljuške nije meren;
  - veza sa suženjem drške je tumačenje i nije proverena;
  - isprobana je samo bilinearna interpolacija; promena veličine slike u pravom softveru kamere nije
    isprobana;
  - trening koji bi rub slike maskirao ili nasumično menjao, da mreža ne zavisi od njega, nije pokušan;
  - nije poznato kako će se ponašati slike pravog senzora, koje se takođe prevzorkuju na 0,2 mm/px.
- **Modeli šuma** (drift, tekstura, ostaci markera) su moje pretpostavke, a ne izmerene vrednosti.
- **Podaci i koraci:** efekat „više podataka“ nije odvojen od „više koraka treninga“. Kontrolni
  trening (20k sa 240 epoha) nije pokrenut.
- **CAD baseline** je moja, jednostavna implementacija. To je donja granica za metode bez učenja,
  a ne jak konkurent.
- **Pričvršćenost viljuške** je idealna: stega nije modelovana.
- **Uslovi pokretanja:**
  - deo treninga je rađen na usporenom GPU-u; to utiče samo na vreme, ne na rezultate;
  - pretraga je renderovana na Intel iGPU-u, gde dubina odstupa od NVIDIA-e u proseku 0,001 mm.
