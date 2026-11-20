# Rezime: lokalizacija viljuške taktilnim senzorom GelSight Mini (simulacija u MuJoCo)

Dokument se čita samostalno, bez README-a. Uz svaki broj u zagradi stoji fajl iz koga je izračunat. Putanje su
relativne prema `results_centre/`, osim kad počinju imenom foldera iz korena projekta (`dataset_mini/`,
`meshes/`). Raniji tok rada (v1 → v3 na standardnom senzoru) opisan je u `REZIME_hronologija.md`.

---

## 1. Cilj

Iz jednog ili više otisaka taktilnog senzora treba odrediti gde je težište viljuške u odnosu na senzor, odnosno
kuda pomeriti senzor. Usput se određuju i deo viljuške pod senzorom, strana viljuške (licem nagore ili nadole) i
njen smer (yaw).

Sve je urađeno u simulaciji, za senzor veličine GelSight Mini. Cilj je da se metod kasnije prenese na pravi
senzor, pa se posebno meri šta bi se pri tom prenosu moglo pokvariti.

## 2. Postavka

- **Simulator** je MuJoCo. Viljuška leži na stolu i pričvršćena je (nema zglobova), a senzor na ruci pritiska
  odozgo (`dataset_mini/metadata.json`, `sim`).
- **Senzor:**
  - polje je 18,6 × 14,4 mm, a slika 186 × 144 px na 0,1 mm/px (`dataset_mini/metadata.json`, `image`, `optics`);
  - tri LED-a (azimut 90°, 210° i 330°, elevacija 35°), markeri na razmaku od 2 mm, šum kamere 1,5 sivih nivoa
    (isti fajl, `optics`).
- **Lanac obrade slike je isti kao kod pravog senzora:**
  - RGB slika → fotometrijski stereo (kalibrisana matrica 3 × 3) → Poissonova integracija → mapa dubine
    (`dataset_mini/metadata.json`, `photometric_calibration`);
  - mreža vidi tu rekonstruisanu dubinu, a ne tačnu dubinu iz simulatora;
  - dubina se smanjuje na 0,2 mm/px (93 × 72 px, INTER_AREA).
- **Viljuška** je CAD model `ForkMilimeter.stl`:
  - dužina je 120,3 mm, masa 9,48 g, a težište na x = −24,3 mm u sistemu viljuške (`dataset_mini/metadata.json`,
    `fork`; granice mreže `meshes/fork_stl/fork_stl_visual.stl`);
  - drška se sužava linearno: širina je 4,0 − 0,05·x mm, tj. 8,0 mm na kraju drške i 5,2 mm kod težišta (preseci
    STL-a, `taper_test/taper_test.py`).
- **Pritisak:** sila je 0,2–10 N, a nagib senzora do 3°. Viljuška se postavlja nasumično u kvadrat ±30 mm i
  nasumično okreće (`dataset_mini/metadata.json`, `config`).

## 3. Skup podataka

- **Veličina:** `dataset_mini/` zauzima 19 GB i ima 100.000 trening, 1.000 validacionih i 5.000 test pritisaka
  (test2) (`dataset_mini/cache_mini.npz`, polje `split`).
- **Test skup** ima 3.473 pritiska na viljušci (zupci 1.367, dlan 603, vrat 548, drška 955), 1.035 na stolu i
  492 bez kontakta (isti fajl).
- **Ponovljivost:** svaki uzorak se iz svog seed-a regeneriše bit-identično.

## 4. Model

- **Mreža** je ResNet enkoder–dekoder sa 9,6 M parametara (prebrojano iz `mini_robust/model_100k_seed0.pt`).
- **Ulaz:** dubina 93 × 72 px i dva koordinatna kanala.
- **Izlazi** (`train_centre.py`):
  - (a) kontakt: nema / sto / viljuška;
  - (b) deo: zupci, dlan, vrat, drška;
  - (c) strana;
  - (d) mapa verovatnoće položaja težišta, 150 × 150 ćelija od 2 mm (±150 mm). Iz nje se uzimaju procena
    (najverovatnija ćelija) i „90 % oblast“, najmanji skup ćelija koji nosi 90 % verovatnoće.
- **Pretraga sa više pritisaka:**
  - mape uzastopnih pritisaka se množe (Bajesovo kombinovanje);
  - sledeći pritisak se bira prema trenutnoj mapi;
  - pretraga staje kad 90 % oblast padne ispod 4 mm² ili posle 10 pritisaka.
- **Konačni model** je `mini_robust` s0, `mini_robust/model_100k_seed0.pt` (`FINAL_MODEL.md`; izbor u odeljku 6.1):
  - trenira se na 100.000 pritisaka, 120 epoha, 4,5 h na GPU (`mini_robust/model_100k_seed0.json`, 16.339 s);
  - uz „robusnu“ augmentaciju (odeljak 5.2);
  - na testu je spoljni pojas od 2 px postavljen na nulu, isto kao u delu treninga.

## 5. Nalazi o robusnosti

### 5.1 Mreža određuje mesto na dršci iz širine drške

- **Problem:** drška je dugačka, ravna traka. Jedan otisak Mini-ja (18,6 mm) često ne vidi ni vrat ni kraj
  drške. Jedini lokalni znak gde je senzor duž drške je njena širina, koja se menja samo 0,05 mm po mm.
- **Test** (`taper_test/taper_mini_robust_s0.json`):
  - korišćeno je 509 test pritisaka iz sredine drške;
  - slika je raširena ili sužena poprečno na osu drške za faktor s;
  - meri se pomak procene duž viljuške;
  - ako mreža čita širinu, očekivani pomak je (s − 1) · širina / 0,05.

  | s | pomak procene (medijana) | očekivano iz geometrije |
  |---|---|---|
  | 0,90 | −12,2 mm | −13,0 mm |
  | 0,95 | −6,3 mm | −6,5 mm |
  | 1,05 | +5,6 mm | +6,5 mm |
  | 1,10 | +10,8 mm | +13,0 mm |

  - Bočni pomak je ispod 0,06 mm.
  - Prethodni model (mini_exact) daje skoro iste brojeve (`taper_test/taper_mini_exact_s0.json`).
- **Zaključak:** mreža meri apsolutnu širinu drške sa tačnošću ispod piksela i iz nje čita mesto duž drške. U
  simulaciji to radi jer je viljuška uvek ista i savršena. Posledice:
  - svaka greška razmere mm/px direktno pomera procenu (5.3);
  - druga viljuška, sa drugačijim suženjem ili paralelnom drškom, ne bi radila bez novog treninga.
- **Jedan pritisak na dršci je zato slab:** p90 greške je 23,3 mm, a na zupcima 1,45 mm
  (`mini_cad/compare_cad_vs_net_mini_robust_s0.json`, čist ulaz, 250 pritisaka po delu). Rešava se tek sa više
  pritisaka (odeljak 6).

### 5.2 Osetljivost na obradu slike i na ivicu slike

- **Šta je pronađeno.** Prethodni model (mini_exact, 3 seed-a) je u pretrazi zavisio od sitnica u obradi
  slike (`mini_robust/diag/criterion.json`, deo `mini_exact`):
  - slika kako je renderovana: konačna greška 0,71 mm, težište u 90 % oblasti u 83 % epizoda;
  - slika prevzorkovana na razmeru 0,99: 0,56 mm i 92 %;
  - samo potamnjen spoljni red piksela: 0,76 mm, pomak duž drške +0,64 mm;
  - razmera 0,99 bez potamnjenog ruba: 1,16 mm, pomak −1,14 mm;
  - raspon pomaka duž drške preko 8 načina obrade je 1,80 mm.
- **Uzrok:** mreža je kao znak koristila i potamneli rub, koji simulirano skaliranje ostavlja na ivici slike.
- **Ispravka u treningu** (`mini_robust/`):
  - nasumično prevzorkovanje: 4 interpolacije, pomak 0–1 px, zum 0,97–1,03;
  - 25 % slika ostaje neprevzorkovano;
  - nasumičan tretman ivice, posebno za svaku stranu;
  - na testu je spoljni pojas od 2 px uvek na nuli.
- **Rezultat ispravke:**
  - svi ulazi na neto razmeri 1 daju istu pretragu: 0,62–0,65 mm, pomak −0,24 do −0,38 mm
    (`mini_robust/diag/criterion.json`);
  - tretman ivice više ništa ne menja (identični brojevi sa potamnjenim rubom i bez njega);
  - jedan pritisak ima p90 2,16–2,19 mm za sve takve ulaze, a mini_exact 1,84–2,28 mm
    (`mini_robust/input_interp.json`, `mini_exact/input_interp.json`).
- **Cena:** na slici kako je renderovana jedan pritisak je lošiji (p90 2,16 prema 1,84 mm, medijana 0,89 prema
  0,88 mm). Pretraga je bolja (0,62 prema 0,71 mm, težište u 90 % oblasti 95 % prema 83 %).
- **Kriterijum nije ispunjen** (`mini_robust/diag/criterion.json`):
  - zahtev je bio raspon < 0,2 mm preko 8 ulaza;
  - izmereno je 1,16 mm (konačna greška) i 2,95 mm (pomak duž drške).
  - Uzrok je razmera (5.3). Usrednjavanje na testu (TTA, 11 varijanti) to ne menja: 1,11 i 2,99 mm
    (`mini_robust_tta/diag/criterion.json`), a košta oko 11 puta više vremena (`runtime_mini/runtime_mini.json`).
  - **Jedina dozvoljena dodatna promena treninga** bilo je fiksno Gausovo zamućenje ulaza (σ = 1 px = 0,2 mm),
    u treningu i na testu (`mini_robust_aa/`). Ni ono ne prolazi: rasponi su 1,30 mm i 2,98 mm
    (`mini_robust_aa/diag/criterion.json`). Očekivano je, jer zamućenje ne menja apsolutnu širinu drške: u testu
    suženja AA se pomera isto (`taper_test/taper_mini_robust_aa_s0.json`, s = 1,05: +5,6 mm).

### 5.3 Zahtev za razmeru mm/px

- **Pretraga:** razmera pomera konačnu procenu duž drške za oko 0,74 mm po 1 % (`mini_robust/diag/criterion.json`):

  | razmera | 0,98 | 0,99 | 1,00 | 1,02 |
  |---|---|---|---|---|
  | pomak duž drške | −1,72 mm | −1,00 mm | −0,24 mm | +1,22 mm |
  | konačna greška (medijana) | 1,78 mm | 1,04 mm | 0,62 mm | 1,29 mm |

  - Kad se pomak oduzme, greška je 0,56–0,66 mm za svaki ulaz, tj. sva razlika je taj pomak.
  - Očekivanje iz suženja (5.1): 1 % od 5,2 mm je 0,052 mm širine, odnosno oko 1 mm duž drške.
- **Jedan pritisak** (p90, test2, `mini_robust/input_scale_sweep_wide.json`):

  | razmera | 0,96 | 0,97 | 0,98 | 0,99 | 1,00 | 1,01 | 1,02 | 1,03 | 1,04 |
  |---|---|---|---|---|---|---|---|---|---|
  | p90 | 5,38 | 4,32 | 3,39 | 2,55 | 2,16 | 2,58 | 3,47 | 4,60 | 5,64 mm |

- **Prividna tolerancija starog modela je bila artefakt.** mini_exact izgleda neosetljiv na smanjenje (0,98 → p90
  1,85 mm, prosek 3 seed-a), ali samo dok je rub potamnjen kao posle simuliranog skaliranja. Sa nepotamnjenim rubom 0,98 daje
  3,07–3,17 mm (`mini_exact/input_edge.json`). Prava kamera sa pogrešnim mm/px nema potamnjen rub.
- **Zahtev:** da bi pomak ostao ispod 0,2 mm, mm/px mora biti poznat na oko ±0,3 %.
- **Kalibracija** (`v2/calibration.json`, 240 ponavljanja u simulaciji, sa driftom i teksturom):
  - etalon sa dve kuglice: 95 % ponavljanja ≤ 0,07 %, najgore 0,12 % → dovoljno;
  - pomeraji robota: 95 % ponavljanja ≤ 0,27 %, najgore 0,42 % → na granici.

### 5.4 Drift osvetljenja je glavni izvor šuma

- **U rekonstruisanoj dubini** drift osvetljenja je daleko najjači efekat (`noise_effect/recon_noise_effect.json`,
  200 pritisaka). Najveća lažna visina po pritisku, medijana (p90):

  | vrsta šuma | lažna visina |
  |---|---|
  | drift 3 % | 0,11 (0,24) mm |
  | drift 6 % | 0,25 (0,55) mm |
  | drift 10 % | 0,58 (1,19) mm |
  | tekstura gela, 2–3 nivoa | 0,01–0,02 mm |
  | ostaci markera | 0,006 mm |

- **Mreža** je trenirana sa driftom i na njega je skoro neosetljiva (`mini_cad/compare_cad_vs_net_mini_robust_s0.json`):
  - drift 10 %: medijana 0,89 mm, p90 2,46 mm;
  - čista slika: 0,87 / 2,14 mm.
- **CAD metod** (odeljak 7) na zupcima pod driftom 10 % pada sa 6,8 na 25,8 mm medijane.
- **Za pravi senzor** ovo znači: referentni kadar treba često obnavljati i drift izmeriti (`measure_noise.py`).

## 6. Rezultati pretrage

300 epizoda, isti počeci za sve modele, čist ulaz (`mini_robust/search/search_metrics.json`, `v2_100k_s0_clean`):

| | konačni model (`mini_robust` s0) | mini_exact, srednja vrednost 3 seed-a (`mini_exact/seeds_100k.json`, `mini_robust/diag/criterion.json`) |
|---|---|---|
| 90 % oblast < 16 mm² | 100 % epizoda, medijana 2 pritiska | 100 %, medijana 2 pritiska |
| za najviše 2 / 3 pritiska | 80 % / 97 % | 95 % / 99,7 % |
| konačna greška, medijana / p90 | **0,62 / 1,18 mm** | 0,71 / 1,26 mm |
| greška > 5 mm | 0 % | 0 % |
| težište u 90 % oblasti na kraju | **95 %** | 83 % |
| 90 % oblast < 4 mm² (rano zaustavljanje) | 49 % | 69 % |

- **Konačni model je tačniji i pošteniji, ali oprezniji:** procena je bliža (0,62 prema 0,71 mm) i 90 % oblast
  češće sadrži težište (95 % prema 83 %). Oblasti su mu šire, pa ređe staje pre 10. pritiska (49 % prema 69 %)
  i ređe dostiže 16 mm² za 2 pritiska (80 % prema 95 %).

- **Po delu prvog pritiska** konačna medijana je 0,60 (zupci), 0,65 (dlan), 0,65 (vrat) i 0,63 mm (drška).
  Dvosmislenost drške se, dakle, rešava sa više pritisaka.
- **Šum** dubine od 0,05 mm: 0,60 mm. **Razmera 0,98:** 1,78 mm, a težište je u 90 % oblasti samo u 21 %
  epizoda (5.3).
- **Smer i strana na kraju pretrage** (`viz/mini_robust_s0/yaw_search.json`): greška smera ima medijanu 0,4° i
  p90 1,4°; smer je okrenut za 180° u 4 % epizoda; strana je tačna u 90,7 %.

### 6.1 Izbor konačnog modela

Pravilo je zapisano pre rezultata (`RUN_LOG.md`, 19.11. u 20:05 i 20:30):
1. najviše ispunjenih delova kriterijuma;
2. kod nerešenog, bolja čista pretraga, ali samo ako upareni bootstrap na istih 300 početaka isključi nulu;
3. inače jeftinije zaključivanje.

- **Kriterijum:** svi kandidati ispunjavaju 1 od 3 dela, samo čistu pretragu (`*/diag/criterion.json`). To su
  mini_exact (3 seed-a), mini_robust s0, mini_robust sa TTA i mini_robust_aa s0.
- **Čista pretraga** (`final_select/*.json`):
  - najbolji je AA, 0,592 mm;
  - AA − robust = −0,028 mm [−0,084, +0,018], dakle nerešeno;
  - AA − TTA = −0,019 mm [−0,079, +0,030], takođe nerešeno;
  - AA − mini_exact je −0,135 / −0,145 / −0,071 mm za seed 0 / 1 / 2, i svi intervali isključuju nulu, pa mini_exact
    otpada.
- **Cena** (`runtime_mini/runtime_mini.json`):
  - robust 1,25 ms (GPU) i 23,4 ms (CPU, 1 nit);
  - AA 1,37 i 23,7 ms;
  - TTA 13,8 i 261 ms.
- **Konačni model je `mini_robust` s0.**
- **Napomena:** AA ima bolji p90 (1,07 prema 1,18 mm) i 0,19 pritisaka manje u proseku
  (`final_select/aa_vs_robust.json`). Pravilo je unapred biralo samo po medijani, pa AA ostaje rezervni model.

## 7. Poređenje sa CAD metodom bez treniranja

- **Metod:** rekonstruisana dubina se poredi sa šablonima iz CAD modela za svaki položaj i ugao (gruba
  korelacija, pa fino podešavanje). Ne treba mu trening. Isti pritisci i isti uslovi za oba metoda.
- **Jedan pritisak** (1.000 pritisaka test2, po 250 po delu, čist ulaz;
  `mini_cad/compare_cad_vs_net_mini_robust_s0.json`):

  | deo | mreža: medijana / p90 | CAD: medijana / p90 | u krugu 5 mm: mreža / CAD |
  |---|---|---|---|
  | zupci | 0,86 / 1,45 mm | 6,82 / 107 mm | 99 % / 46 % |
  | dlan | 0,83 / 1,26 mm | 1,78 / 73 mm | 98 % / 66 % |
  | vrat | 0,80 / 1,94 mm | 4,70 / 24 mm | 96 % / 52 % |
  | drška | 1,07 / 23,3 mm | 7,03 / 56 mm | 82 % / 37 % |
  | sve | 0,87 / 2,14 mm | 4,96 / 74 mm | 94 % / 50 % |

- **Kalibracija (pokrivenost 50 / 80 / 95 % oblasti):**
  - mreža 0,85 / 0,97 / 0,99;
  - CAD 0,70 / 0,88 / 0,97, ali njegova 90 % oblast ima medijanu oko 12.000 mm², tj. skoro ništa ne kaže.
- **Pretraga** (`mini_cad/search/search_metrics.json`):

  | prvi pritisak | CAD: konačna greška | CAD: greška > 5 mm | mreža: konačna greška |
  |---|---|---|---|
  | zupci | 3,60 mm | 34 % | 0,60 mm |
  | dlan | 2,81 mm | 17 % | 0,65 mm |
  | vrat | 6,20 mm | 56 % | 0,65 mm |
  | drška | 5,41 mm | 58 % | 0,63 mm |
  | sve | 4,09 / 11,4 mm | 41 % | 0,62 / 1,18 mm |

  - CAD dostiže 90 % oblast < 16 mm² u samo 29 % epizoda i uvek troši svih 10 pritisaka.
  - Mreža to dostiže u 100 % epizoda, za medijanu od 2 pritiska.
- **Vreme po pritisku** (`runtime_mini/runtime_mini.json`, mašina bez drugog opterećenja):
  - mreža: 1,25 ms na GPU, 5,45 ms na CPU sa 8 niti, 23 ms na CPU sa 1 niti;
  - CAD: 38 ms, 449 ms i 1,33 s.
  - Mreža je 31–82 puta brža (GPU 31×, CPU sa 1 niti 57×, CPU sa 8 niti 82×).
- **Uočeno:** CAD je bolji na slici smanjenoj za 2–5 % (pretraga na 0,98: 3,02 mm). Verovatno njegovi šabloni
  i rekonstruisana dubina nisu iste razmere. To nije istraživano.

## 8. Preglednik i video

- **Interaktivni preglednik** (`search_viewer.py`, model iz `FINAL_MODEL.md` ili `--net`; demo: `prezentacija/demo.sh`):
  - zelena „senka“ viljuške prikazuje procenjenu pozu (težište, smer, strana) preko prave viljuške;
  - zelena i crvena kugla su procenjeno i pravo težište; 90 % oblast je prikazana kao pločice na stolu;
  - prikazuju se i tačke pritisaka, taktilna slika, mapa verovanja i tekstualni panel;
  - tasteri: F6 jedan korak pretrage, F7 do kraja, F8 brisanje verovanja, F9 nova poza viljuške;
  - gore desno je pogled odozgo: prava viljuška sivo, viljuška u proceni zeleno, uz tabelu procene i istine
    (x, y, ugao, strana) i greške;
  - bočni meniji MuJoCo-a su na startu sakriveni. Preglednik i sam obrađuje F6 (menja prikaz osa), pa se to posle
    svakog tastera vraća.
- **Procena smera iz jednog pritiska** (`viz/mini_robust_s0/yaw_heuristic.json`, 3.473 pritiska):
  - medijana greške je 4,4° na zupcima, 3,8° na dlanu, 2,1° na vratu i 1,5° na dršci;
  - na dršci se u 25 % pritisaka pogreši kraj na kom je glava (okret za 180°), iako je osa tačna (medijana
    1,1°);
  - sve medijane su ispod 10°, pa posebna glava mreže za smer nije bila potrebna.
- **Provere:**
  - koraci u pregledniku ponavljaju sačuvane epizode tačno (20 od 20, `viz/mini_robust_s0/check_replay.log`);
  - test bez prozora prolazi (`viz/mini_robust_s0/viewer_selftest.png`).
- **Video** (`videos/mini_robust_s0/`, `index.json`):
  - 9 epizoda kao mp4 sa ključnim kadrovima, i sve kao gif (`videos/mini_robust_s0_gif/`);
  - pokriveno je licem nadole i nagore × prvi pritisak na zupcima, dlanu, vratu i dršci, plus najgora epizoda
    (start 837: 2,18 mm, pogrešna strana);
  - u ostalih 8 epizoda konačna greška je 0,21–1,63 mm.

## 9. Poznata ograničenja

- **Samo simulacija.** Pravi senzor i robot nisu korišćeni.
- **Jedna viljuška, pričvršćena.** Model koristi tačno suženje ove drške (5.1). Druga viljuška zahteva nove
  podatke.
- **Razmera:** greška mm/px od 1 % pomera rezultat za oko 0,7 mm (5.3). Kriterijum nezavisnosti od obrade
  slike nije ispunjen.
- **Jedan seed:** konačni model je treniran jednom. Rasipanje između seed-ova kod sličnog modela (mini_exact) je
  std p90 0,07 mm (`mini_exact/seeds_100k.json`).
- **Gel pravog Mini-ja** (debljina 4,25 mm, mekoća) nije modelovan; fotometrijska kalibracija je simulirana.
- **Modeli šuma** (drift, tekstura, ostaci markera) su pretpostavke, a ne merenja.
- **Prevzorkovanje pravog kadra** (faktor oko 3,4 umesto 2) još nijedna skripta ne radi.
- **CAD baseline** je jednostavna sopstvena implementacija, donja granica za metode bez učenja.
- **Interaktivni prozor preglednika** otvoren je samo na probi. Ispravka za F6 i novi panel provereni su samo bez
  prozora.

## 10. Šta je potrebno za pravi senzor

1. **Kalibracija mm/px** na ±0,3 % ili bolje, etalonom sa dve kuglice (`calibrate_real.py`). To je
   najvažniji pojedinačni zahtev.
2. **Ista obrada slike** kao u simulatoru, korak po korak (tabela u README, „Real-sensor tooling“):
   - referentni kadar;
   - fotometrijska kalibracija loptom na izmerenom mm/px;
   - dubina istom funkcijom;
   - smanjenje na 0,2 mm/px sa INTER_AREA na 93 × 72 px;
   - učitavanje modela kroz `centre_v3.load_net`, koje samo primenjuje maskiranje ivice.
3. **Merenje šuma** (`measure_noise.py`), naročito drifta osvetljenja. Augmentaciju treba podesiti na izmerene
   vrednosti, a referentni kadar često obnavljati.
4. **Provera suženja na pravoj viljušci:** širina drške mora odgovarati CAD-u, inače mesto na dršci neće biti
   tačno.
5. **Prvi test na hardveru:** nekoliko pritisaka na poznatim mestima, da se proveri da li se procena pomera
   duž drške (znak greške razmere ili oblika).
