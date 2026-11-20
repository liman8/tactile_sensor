# Konačni model (FINAL)

- **path**: `results_centre/mini_robust/model_100k_seed0.pt`
- **SHA-256**: `01dd329c9a790c9563d35bbee5da8982760e970846a28de56a6c18cec82582d8`
- **Izabran**: 2026-11-20, po pravilu zapisanom unapred (`RUN_LOG.md`, 2026-11-19 20:05 i 20:30).

Putanje su relativne prema korenu projekta. Uz svaki broj stoji fajl iz koga je uzet; fajlovi su u `results_centre/`,
osim ako nije drugačije navedeno. `search_viewer.py` čita putanju iz reda `path` ovog fajla.

## 1. Recept

- **Senzor:** polje GelSight Mini, 18,6 × 14,4 mm. Slika je 186 × 144 px na 0,1 mm/px, a ulaz mreže 93 × 72 px na
  0,2 mm/px.
- **Podaci:** `dataset_mini/`, 100.000 trening pritisaka (`train2`), 1.000 validacionih; seed 0.
- **Mreža:** ResNet enkoder i dekoder mape težišta (`train_centre.build_model`), 9.593.882 parametra. Izlazi su
  kontakt, deo, strana i mapa 150 × 150 ćelija od 2 mm.
- **Trening:**
  - 120 epoha, najbolja epoha 119, trajanje 16.339 s (`mini_robust/model_100k_seed0.json`);
  - AdamW, one-cycle raspored (maks. 2·10⁻³), batch 64, bf16.
- **Augmentacija** `CENTRE_AUG_ROBUST=1` (`centre_v2.augment_robust`):
  - prevzorkovanje sa 4 vrste interpolacije, pomakom 0–1 px i zumom 0,97–1,03;
  - 25 % ulaza ostaje neprevzorkovano;
  - nasumičan tretman svake ivice posebno;
  - pored toga rotacija i ogledalo, šum, zamućenje, drift osvetljenja i tekstura gela.
- **Komanda:**
  `GELSIGHT_SENSOR=mini CENTRE_AUG_ROBUST=1 CENTRE_MINI_OUT=results_centre/mini_robust python centre_v3.py train --size 100000 --seed 0`

## 2. Zaključivanje

1. **Ulaz:** dubina rekonstruisana iz RGB slike (fotometrijski stereo i Poissonova integracija), smanjena na 0,2 mm/px
   sa INTER_AREA, plus koordinatni kanali x i y senzora.
2. **Učitavanje** obavezno kroz `centre_v3.load_net(path, device)`. Ono postavlja spoljni pojas od 2 px na nulu
   (`test_border_px`), isto kao u treningu. Bez TTA i bez zamućenja ulaza.
3. **Procena jednog pritiska:** najverovatnija ćelija mape; 90 % oblast je najmanji skup ćelija sa 90 % verovatnoće.
4. **Pretraga (v2):**
   - mape se prebacuju u koordinate stola i množe, uz 1 % uniformne raspodele;
   - sledeći pritisak bira se po očekivanom dobitku informacije, a posle praznog pritiska senzor se vraća uz poslednji
     kontakt;
   - pretraga staje kad je 90 % oblast < 4 mm² ili posle 10 pritisaka.
5. **Vreme po pritisku**, samo mreža (`runtime_mini/runtime_mini.json`): 1,25 ms na GPU, 5,45 ms na CPU sa 8 niti,
   23,4 ms na CPU sa 1 niti. U simulaciji, sa fizikom i renderovanjem, pritisak traje 0,72 s
   (`mini_robust/search/search_metrics.json`).

## 3. Zašto ovaj model

**Kriterijum:** svaki kandidat ispunjava 1 od 3 dela (`*/diag/criterion.json`).

| kandidat | raspon konačne greške (< 0,2) | raspon duž drške (< 0,2) | čista pretraga prema mini_exact (≤ +0,1) |
|---|---|---|---|
| mini_exact (3 seed-a) | 0,59 mm ✗ | 1,80 mm ✗ | 0 ✓ |
| **mini_robust s0** | 1,17 mm ✗ | 2,95 mm ✗ | −0,09 ✓ |
| mini_robust + TTA | 1,11 mm ✗ | 2,99 mm ✗ | −0,10 ✓ |
| mini_robust_aa s0 | 1,30 mm ✗ | 2,98 mm ✗ | −0,12 ✓ |

**Nerešeno → čista pretraga.** Medijana konačne greške se poredi uparenim bootstrap-om na istih 300 početaka
(`final_select/*.json`):
- najbolji je AA sa 0,592 mm;
- AA − robust = −0,028 mm [−0,084, +0,018] i AA − TTA = −0,019 [−0,079, +0,030]: oba intervala obuhvataju 0, pa je
  nerešeno;
- AA − mini_exact je −0,135 / −0,145 / −0,071 mm za seed 0 / 1 / 2, a svi intervali isključuju 0, pa mini_exact
  otpada.

**Nerešeno među AA, robust i TTA → jeftinije zaključivanje** (`runtime_mini/runtime_mini.json`):

| | CPU, 1 nit | CPU, 8 niti | GPU |
|---|---|---|---|
| mini_robust | 23,43 ms | 5,45 ms | 1,25 ms |
| mini_robust_aa | 23,67 ms | 5,57 ms | 1,37 ms |
| mini_robust + TTA | 260,7 ms | 62,1 ms | 13,8 ms |

Robust je najjeftiniji u sva tri slučaja.

**Napomena uz izbor:** AA ima bolji p90 konačne greške (1,07 prema 1,18 mm; CI razlike [−0,241, +0,000]) i u
proseku 0,19 pritisaka manje (CI [−0,32, −0,05]) (`final_select/aa_vs_robust.json`). Pravilo je unapred biralo
samo po medijani. AA je zato ostao rezervni kandidat, a ne konačni model.

## 4. Rezultati

### 4.1 Jedan pritisak

- **Mini test2, ceo skup, slika kako je renderovana** (`mini_robust/input_interp.json`): medijana 0,89 mm, p90 2,16 mm.
- **Po delu,** 1.000 pritisaka sa po 250 na svakom delu, čist ulaz (`mini_cad/compare_cad_vs_net_mini_robust_s0.json`):

| deo | medijana | p90 | unutar 5 mm |
|---|---|---|---|
| zupci | 0,86 mm | 1,45 mm | 99 % |
| dlan | 0,83 mm | 1,26 mm | 98 % |
| vrat | 0,80 mm | 1,94 mm | 96 % |
| drška | 1,07 mm | 23,3 mm | 82 % |

- **Smer i strana iz jednog pritiska** (`viz/mini_robust_s0/yaw_heuristic.json`): medijana greške smera je 1,5–4,4°
  po delu. Na dršci je smer okrenut za 180° u 25 % pritisaka, iako je osa tačna.

### 4.2 Pretraga

300 epizoda, čist ulaz (`mini_robust/search/search_metrics.json`):

- **90 % oblast < 16 mm²:** u 100 % epizoda, sa medijanom od 2 pritiska; za najviše 2 pritiska u 80 % epizoda, za
  najviše 3 u 97 %.
- **Konačna greška:** medijana 0,62 mm, p90 1,18 mm; greška veća od 5 mm u 0 % epizoda.
- **Pravo težište u konačnoj 90 % oblasti:** 94,7 %.
- **Rano zaustavljanje (< 4 mm²):** 49 % epizoda; prosečno 9,24 pritiska.
- **Uslovi:** šum 0,05 mm daje 0,60 mm. Razmera 0,98 daje 1,78 mm, a težište je u 90 % oblasti u samo 21 %
  epizoda (`mini_robust/diag/criterion.json`).
- **Smer i strana na kraju** (`viz/mini_robust_s0/yaw_search.json`): medijana 0,4°, p90 1,4°; okrenut u 4 %
  epizoda; strana tačna u 90,7 %.

### 4.3 Prema CAD metodu bez učenja

Isti pritisci i isti uslovi (`mini_cad/compare_cad_vs_net_mini_robust_s0.json`, `mini_cad/search/search_metrics.json`,
`runtime_mini/runtime_mini.json`):

- **Jedan pritisak, medijana:** mreža 0,87 mm, CAD 4,96 mm.
- **Pretraga:** konačna greška 0,62 prema 4,09 mm; greška veća od 5 mm u 0 % prema 41 % epizoda.
- **Vreme:** mreža 1,25 ms prema 38 ms CAD-a na GPU-u, odnosno 23 ms prema 1,33 s na CPU-u sa jednom niti.

## 5. Poznata ograničenja

- **Razmera mm/px:**
  - 1 % greške pomera procenu duž drške za oko 0,74 mm (`mini_robust/diag/criterion.json`);
  - zato mm/px mora biti poznat na oko ±0,3 %;
  - kriterijum nezavisnosti od obrade slike nije ispunjen.
- **Suženje drške:** mreža čita mesto na dršci iz širine drške (`taper_test/taper_mini_robust_s0.json`). Druga
  viljuška zahteva novi trening.
- **Jedan seed.** Kod sličnog modela (mini_exact) rasipanje p90 između seed-ova je 0,07 mm
  (`mini_exact/seeds_100k.json`).
- **Samo simulacija:**
  - gel pravog Mini-ja nije modelovan;
  - modeli šuma su pretpostavke;
  - viljuška je pričvršćena.
- **Okretanje smera:** u 12 % epizoda koje počinju na dršci smer ostaje okrenut za 180° do kraja pretrage
  (`viz/mini_robust_s0/yaw_search.json`).
- **Rano zaustavljanje** je ređe nego kod mini_exact (49 % prema 69 %). Model je oprezniji, a njegove oblasti su
  šire.
