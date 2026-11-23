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

### 6.2 Rasipanje između seed-ova konačnog recepta

Konačni recept (`mini_robust`) je posle izbora istreniran još dva puta, sa seed-om 1 i 2. Sve ostalo je isto:
120 epoha, oko 3,1 h po seed-u. Seed-ovi su testirani na istih 300 početaka pretrage i na istom test skupu
(`mini_robust/seeds_100k.json`, `mini_robust/diag/seeds_summary.json`; README „v3, Mini“, odeljak 12).

| | s0 (konačni) | s1 | s2 | srednja vrednost ± std | mini_exact, 3 seed-a |
|---|---|---|---|---|---|
| jedan pritisak, medijana / p90 | 0,894 / 2,16 mm | 0,883 / 2,08 mm | 0,885 / 1,97 mm | 0,89 / 2,07 ± 0,10 mm | 0,88 / 1,84 ± 0,07 mm |
| pretraga, konačna greška, medijana | 0,620 mm | 0,592 mm | 0,599 mm | **0,603 ± 0,015 mm** | 0,708 ± 0,040 mm |
| pretraga, konačna greška, p90 | 1,18 mm | 1,14 mm | 1,09 mm | 1,14 ± 0,04 mm | 1,26 ± 0,11 mm |
| težište u 90 % oblasti na kraju | 95 % | 93 % | 93 % | 94 ± 1 % | 83 ± 4 % |
| razmera 0,98 / 1,02: pomak duž drške | −1,72 / +1,22 mm | −1,76 / +1,17 mm | −1,72 / +1,20 mm | −1,74 / +1,20 mm | −0,30 / +0,66 mm |
| ispunjeni delovi kriterijuma | 1 od 3 | 1 od 3 | 1 od 3 | | po 1 od 3 |

- **Recept je stabilan.** Konačna greška pretrage se između seed-ova razlikuje za 0,015 mm (std), a p90 jednog
  pritiska za 0,10 mm, slično kao kod mini_exact (0,07 mm).
- **Pad kriterijuma je osobina recepta, a ne seed-a 0.** Sva tri seed-a padaju na ista dva dela, sa skoro istim
  rasponom (1,12–1,16 i 2,92–2,95 mm). Zavisnost od stvarne razmere je ista na svakom seed-u (5.3).
- **Konačni model (seed 0) je najslabiji od tri, ali malo.** U pretrazi ima 0,620 prema 0,592 i 0,599 mm, a
  ima i najveću validacionu grešku. Njegovi brojevi zato malo potcenjuju recept, za oko 0,02 mm.
  - Konačni model se ne menja. Pravilo ga je izabralo pre nego što su ovi seed-ovi postojali.
  - Izbor najboljeg od tri seed-a po test brojevima bio bi izbor na test skupu.
- **Prema mini_exact, sa po 3 seed-a:** čista pretraga je bolja za 0,105 mm.
  - Bootstrap po počecima daje [−0,152, −0,056].
  - Welch test na nivou seed-ova (3 prema 3) daje p = 0,033.
  - Ranije, sa samo jednim seed-om robust modela, razlika prema mini_exact s2 nije bila značajna.
- **Rezervni AA (jedan seed) je unutar rasipanja recepta.** Njegova medijana (0,592 mm) jednaka je medijani seed-a 1.
  To potvrđuje da je razlika AA − robust s0 iz 6.1 nerešena.

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
- **Seed-ovi:** konačni recept je istreniran 3 puta (6.2). Rasipanje je malo, a seed 0 je za oko 0,02 mm
  slabiji od proseka. Rezervni model (AA) ima samo jedan seed. Test suženja, poređenje sa CAD-om, merenje smera,
  preglednik i videi rađeni su samo sa seed-om 0.
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

## 11. Više predmeta: koji je predmet i kakva mu je poza (`results_multi/`)

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
- Glave: kontakt (lebdi / sto / predmet), identitet predmeta, delovi (poseban blok za svaki predmet), yaw i stabilna
  poza (11.5) i mapa verovatnoće težišta. **Mapa je jedna, zajednička za sve predmete, ali uslovljena hipotezom o
  predmetu** (naučeni vektor predmeta dodaje se ulazu dekodera), pa pretraga dobija posebnu mapu za svaku hipotezu.
  Jedna neuslovljena mapa bi na zajedničkoj dršci pomešala težišta sva četiri predmeta.

### 11.4 Prva pretraga sa nepoznatim predmetom (zamenjena pretragom pune poze, 11.5)

Ovo je bila pretraga pre dodatka o pozi (samo predmet i težište). Kod je sačuvan; svi rezultati u 11.6 su iz
pretrage pune poze (`pose_search.py`).

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

**Test opterećenja pretrage** (model A, istih 300 početaka; pretraga veruje zadatoj pozi ruke, nominalnom CAD-u i
senzoru iz treninga): ADD < 5 mm i prepoznavanje se ne menjaju ni pod jednom smetnjom. Greška težišta raste sa greškom
ruke: šum 0,5 mm / 1° +0,3-0,4 mm, pomeraj kalibracije 1 mm / 1° +0,6-0,8 mm (dodirom se ne vidi), predmet 1 % različit
od CAD-a +0,1-0,25 mm, šum slike ništa; sve zajedno 1,2-1,3 mm. Verovanje tada postaje presigurno (pravo težište u 90 %
oblasti u 58-78 % umesto 88-99 %), jer greška ruke još nije u modelu verovatnoće.

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

### 11.7 Preglednik, demo i video

- `bash prezentacija/demo.sh 6`: model A, predmet je nepoznat pretrazi; Page Down / Page Up menjaju predmet. Senka je
  najverovatniji predmet u proceni pune poze; u tekstu su verovatnoće predmeta, stabilna poza, yaw, greške i ADD.
- `demo.sh 7`: model B na Fork.stl, kako je planirano (obično je nazove ForkMilimeter). `demo.sh 8`: isto, sa visinom
  dodira i poravnatom CAD hipotezom (naknadno). Modeli nisu u git-u: `bash get_models.sh --multi --from FOLDER`.
- Video zapisi su u `results_multi/videos/` (ključni kadrovi u `frames/`), između ostalog ista epizoda kašike bez visine
  (posle 10 pritisaka kaže nož) i sa visinom (kašika posle 4 pritiska). Prozor pregledača u ovom radu nije otvaran;
  provereno je bez prozora da senke ne menjaju taktilne slike i da se sačuvane epizode ponavljaju tačno.

### 11.8 Ograničenja (više predmeta)

- Glave mreže se množe kao nezavisne, a sve vide istu sliku.
- Predviđanje visine iz CAD-a ne uključuje kućište senzora.
- Visina dodira je simulirana tačno, uz pretpostavljeni šum od 0,3 mm; na pravom robotu dodir treba prepoznati po
  pragu sile, a visinu stola kalibrisati.
- Temperatura završne dorade i poravnanje CAD hipoteze odlučeni su posle pogleda na rezultate; to je zapisano.
- Slike zavise od EGL uređaja (podaci na NVIDIA, pretrage na Intel iGPU).
- Sve je simulacija; nijedan pravi senzor nije korišćen.
