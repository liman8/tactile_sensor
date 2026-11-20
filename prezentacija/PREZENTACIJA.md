# Prezentacija: gde je viljuška, samo na dodir (GelSight Mini u simulatoru MuJoCo)

Tekst za izlaganje gostujućim profesorima: šta se prikazuje, šta se kaže i odgovori na verovatna pitanja.

- **Trajanje:** oko 20 min izlaganja, 5–10 min demo, zatim pitanja.
- **Slike i videi:** u `prezentacija/slike/` i `prezentacija/video/`, numerisani redom izlaganja.
- **Izvori brojeva:** svaki broj potiče iz sačuvanog fajla; spisak je na kraju (odeljak 14).
- **Konačni model:** stanje na dan 20.11.2026. Trenutni kandidat je `mini_robust` (seed 0). Poslednji dozvoljeni
  pokušaj poboljšanja (`mini_robust_aa`) se još trenira, pa konačni izbor nije zapisan.

---

## 0. Pre prezentacije: proba (10 min, obavezno)

Interaktivni prozor MuJoCo-a nikad nije otvaran tokom rada; proveren je samo isti kod bez prozora. Zato ga treba
probati na računaru na kom se prezentuje.

1. `bash prezentacija/demo.sh 2`
   - mora da se otvori prozor sa viljuškom i senzorom;
   - pritisni **F6**: senzor se spušta i pritiska dršku;
   - dole levo se pojave taktilna slika i mapa, a gore desno pogled odozgo sa procenjenom viljuškom i brojevima;
   - bočni meniji MuJoCo-a su na startu sakriveni (Tab / Shift+Tab ih prikazuju);
   - posle F6 ne smeju da se pojave ose x, y, z (crvene, zelene i plave strelice). Ako se ipak pojave: Tab →
     Rendering → Frame → None;
   - na laptopu F6–F9 često traže i taster **Fn**.
2. `bash prezentacija/demo.sh 1`
   - u desnom panelu otvori **Control** i pomeri klizač **press** na oko 3;
   - senzor se spušta na zupce, a dole levo piše „FORK p=0.95, 4 tines“.
3. **Ako prozor ne radi** (npr. problem sa GLFW pod Wayland-om), sve postoji kao video u `prezentacija/video/`.
   Isti tok je snimljen bez prozora.
4. Zatvaranje: dugme X na prozoru.

Provera bez prozora prolazi (`bash prezentacija/demo.sh t`): posle F7 pretraga staje posle 7 pritisaka, greška težišta
je 0,11 mm, strana je tačna (`prezentacija/selftest_002235.png`, sa novim panelom gore desno).

---

## 1. Uvod (1 min)

**Reci:**
- Robot treba da nađe viljušku koja leži na stolu samo dodirom, bez kamere koja gleda scenu. Posebno ga zanima
  njeno **težište**, jer ga tamo treba uhvatiti.
- Senzor je **GelSight Mini**: elastični jastučić sa kamerom iza njega. Jedan otisak vidi samo 18,6 × 14,4 mm, a
  viljuška je dugačka 120 mm. Iz malog komadića treba zaključiti gde je ostatak.
- Rad je ceo u simulaciji, ali tako da se isti lanac obrade može primeniti na pravi senzor.

**Pokaži:** `slike/01_scena_senzor_i_viljuska.png` ili `video/00_senzor_pritiska_viljusku.gif`.

---

## 2. Kako radi GelSight (1 min)

**Reci:**
- Gel ima reflektujuću kožu. Ispod je kamera, a sa strane tri LED-a u bojama (crvena, zelena, plava), pod uglom.
- Kad predmet utisne gel, nagib površine menja boju piksela. Svaka boja „vidi“ nagib iz svog pravca.
- Iz boje se računa nagib (fotometrijski stereo), a iz nagiba integracijom visina. Dobija se mapa dubine otiska
  sa rezolucijom ispod milimetra.
- Crne tačke su markeri na gelu. Pomeraju se sa gelom; ovde se samo uklanjaju pre rekonstrukcije.

**Pokaži:** `slike/03_taktilne_slike_i_rekonstruisana_dubina.png`. Gornji red je RGB slika senzora, donji red je
dubina rekonstruisana iz nje.

---

## 3. Kako je modelovano u MuJoCo-u (4 min)

### 3.1 Scena

- Sto, viljuška i senzor na „ruci“ sa 6 aktuatora: x, y, sila pritiska, yaw, pitch i roll.
- Sila je aktuator sile. Senzor sile u gelu očitava zadatu silu (`gel_force`).
- Viljuška leži na stolu u prirodnom položaju (licem nagore ili nadole) i pričvršćena je, ne pomera se.
- Postoji i režim gde se predmet može pomeriti (freejoint), ali on nije korišćen za trening.

### 3.2 Viljuška iz CAD modela

- `ForkMilimeter.stl`: 120,3 mm, 986 trouglova, zatvorena mreža; masa 9,48 g (čelik), težište na x = −24,3 mm.
- **Dve predstave istog tela:**
  - originalna mreža služi za prikaz i za taktilnu sliku;
  - za sudare služi 17 konveksnih delova (CoACD). MuJoCo sudara konveksne ljuske, pa bi jedna ljuska popunila
    razmak između zubaca.
- **Provera:** na strani koju gel dodiruje kolizioni oblik odstupa od CAD-a najviše 22 µm (p99: 6 µm).

**Pokaži:** `slike/02_viljuska_cad_prikaz_i_sudari.png` (boje = konveksni delovi).

### 3.3 Senzor: od fizike do slike

Lanac je u `gelsight_sim.py`:

1. **Fizika kontakta** (MuJoCo, meki kontakt): predmet utone u gel.
2. **Dubina:** ortografska kamera iznutra senzora renderuje samo aktivni predmet. Iz toga se dobija koliko je
   predmet prošao kroz nedeformisanu površinu gela.
3. **Model gela:** elastični „obod“ oko kontakta (Gaussovo širenje, σ = 0,45 mm) i zaobljenje oštrih CAD ivica
   (σ = 0,08 mm).
4. **Fotometrija:** Lambertova koža; tri LED-a na azimutima 90°, 210° i 330°, na 35° elevacije; neravnomerno
   osvetljenje.
5. **Markeri i šum:** markeri na razmaku od 2 mm, koji se pomeraju sa gelom; zamućenje optike 0,6 px; šum kamere
   1,5 nivoa sive.

**Detalj koji vredi pomenuti:** MuJoCo meki kontakt zavisi od mase tela u kontaktu. Masa ruke je zato podešena
tako da gel na istu silu daje isto utiskivanje kao u prvobitnom modelu, sa odstupanjem od −4,5 % do +5 %.

**Senzor Mini:** polje 18,6 × 14,4 mm, 186 × 144 px na 0,1 mm/px. To je tačan isečak sredine standardnog polja
od 32 × 24 mm.

### 3.4 Od slike do dubine: isti lanac kao na pravom senzoru

1. Referentni kadar bez kontakta i razlika od njega; markeri se uklanjaju.
2. **Kalibracija kuglicom poznatog prečnika:** linearna mapa 3 × 3 iz nagiba površine u promenu boje, metodom
   najmanjih kvadrata.
3. Obrnuta mapa daje nagib u svakom pikselu. Poissonova integracija (Neumannov rub) daje visinu.

**Zašto mreža vidi rekonstruisanu, a ne tačnu dubinu:** pravi senzor daje samo rekonstruisanu. Rekonstrukcija
greši:
- na viljušci je otisak preduboko procenjen, +0,09 do +0,17 mm, a vrh za 27–31 %;
- ravan sto se skoro ne vidi, jer ravna površina nema nagib.

Mreža treba da nauči sa tim greškama.

---

## 4. Prvi korak: klasični detektor „viljuška ili ne“ (1 min, demo 1)

- **Metod:** rekonstruisana dubina → ručno birana obeležja (broj i ugao zubaca, oblik) → RandomForest.
- **Rezultati:**
  - 98,3 % tačnosti u novom režimu, bez ponovnog treniranja;
  - CAD viljušku prepoznaje u 99,3 % slučajeva;
  - cena su lažne „viljuške“ na češlju (2,7 %; na galeriji se vidi jedan takav slučaj).
- **Ograničenje, i motivacija za ostatak rada:** kad senzor pritisne samo dršku, detektor kaže „nije viljuška“.
  Drška je samo traka. Treba nešto što zna **koji deo** viljuške dodiruje i **gde je težište** u odnosu na taj deo.

**Pokaži:** `slike/04_rucni_pritisak_zupci_drska_podignut.png` (zupci: FORK, 4 zupca; drška: nije viljuška;
podignut: nema kontakta) ili uživo `demo.sh 1`.

---

## 5. Skup podataka (1 min)

- **100.000 trening, 1.000 validacionih i 5.000 test pritisaka** sa senzorom Mini, 19 GB.
- **Test skup** ima 3.473 pritiska na viljušci: zupci 1.367, dlan 603, vrat 548, drška 955.
- **Svaki pritisak:**
  - viljuška u stabilnom položaju (licem nagore ili nadole), na nasumičnom mestu (±30 mm) i pod nasumičnim uglom;
  - sila od 0,2 do 10 N, nagib senzora do 3°.
- **Oznake su besplatne i tačne:** pravo težište u sistemu senzora, deo viljuške, strana.
- **Ponovljivost:** svaki uzorak se iz svog seed-a regeneriše bit-identično.
- **Provere:**
  - oznake se poklapaju sa slikama (IoU 0,993);
  - zbir kontaktnih sila jednak je očitavanju senzora;
  - trening i test se ne preklapaju.

---

## 6. Model: šta je i zašto baš takav (4 min)

### 6.1 Arhitektura

- **Ulaz:** rekonstruisana dubina smanjena na 0,2 mm/px (93 × 72 px) i **dva koordinatna kanala** (x i y svakog
  piksela u sistemu senzora).
- **Telo:** ResNet enkoder, 10 rezidualnih blokova, 32–256 kanala, 4 stepena sa korakom 2. Veličina je otprilike
  kao ResNet-18.
- **Izlazi (četiri glave):**
  - (a) kontakt: nema / sto / viljuška;
  - (b) deo: zupci, dlan, vrat, drška;
  - (c) strana: licem nagore ili nadole;
  - (d) **mapa verovatnoće položaja težišta**, 150 × 150 ćelija od 2 mm (±150 mm oko senzora).
- **Ukupno 9,6 M parametara.**

### 6.2 Zašto mapa verovatnoće, a ne direktna regresija (x, y)

Ovo je ključna odluka.

- **Dvosmislenost je stvarna.** Otisak sredine drške izgleda skoro isto na celoj dužini drške. Regresija bi dala
  prosek, tj. tačku koja je verovatno pogrešna. Mapa može da prikaže traku duž drške, dva vrha ili luk.
- **Proverljive verovatnoće.** Mapa se trenira unakrsnom entropijom, koja je „pravilno bodovanje“
  (proper scoring rule). Zato se može proveriti da li „90 % oblast“ zaista sadrži težište u 90 % slučajeva.
- **Kombinovanje pritisaka.** Mape iz više pritisaka se samo pomnože (Bajes). Na tome počiva aktivna pretraga.
- **Zašto ne mešavina Gausovih raspodela:** broj modova mora unapred da se zada, modovi se često stapaju ili
  varijanse eksplodiraju, a oblasti treba numerički integrisati.
- **Cena:** kvantizacija od 2 mm i veći izlazni sloj.

### 6.3 Zašto koordinatni kanali i „ravnanje“ umesto usrednjavanja

Konvolucija je neosetljiva na pomeraj, a ovde je **mesto** kontakta u slici upravo ono što određuje gde je
težište u odnosu na senzor. Koordinatni kanali i ravnanje mape obeležja (bez globalnog usrednjavanja) čuvaju tu
informaciju.

### 6.4 Zašto ResNet ove veličine

- Standardna, proverena arhitektura, dovoljno mala za 100k uzoraka. Zaključivanje traje milisekunde: za
  standardni senzor 1,6 ms na GPU i 69 ms na jednoj CPU niti. Za Mini model vreme nije posebno mereno, a ulaz mu
  je manji.
- **Kapacitet nije usko grlo.** Preostala greška je fizička dvosmislenost (drška) i razmera slike (odeljak 9).
  Više podataka tu ne pomaže: Mini na 40k ima p90 na dršci 13,7 mm, a na 100k 13,1 mm.
- **Iskreno:** druge arhitekture nisu sistematski poređene.

### 6.5 Trening

- AdamW, one-cycle raspored (maks. 2·10⁻³), batch 64, 120 epoha; model se bira po validaciji.
- 100k pritisaka traje oko **4,5 h na GPU** (16.339 s).
- **Augmentacija:**
  - rotacija i ogledalo, uz istu transformaciju cilja (viljuška je simetrična po dužoj osi);
  - šum, zamućenje, drift osvetljenja, tekstura gela;
  - „robusno“ prevzorkovanje (odeljak 9.2).

**Pokaži:**
- `slike/05_ulaz_mreze_i_mapa_tezista.png`: levo ulaz, desno mapa. Žuti + je procena, crveni x pravo težište.
  Slika je iz prve verzije sa standardnim senzorom, ali princip je isti.
- `slike/06_augmentacija_robust.png`: šta mreža vidi u treningu.

---

## 7. Pretraga sa više pritisaka (3 min, demo 2)

### 7.1 Petlja

1. Pritisak → rekonstrukcija → mapa težišta.
2. Mapa se prebacuje u koordinate stola, jer je poza senzora poznata.
3. Množi se sa dosadašnjom mapom.
   - Svaka mapa se meša sa 1 % uniformne raspodele, da jedan pogrešan pritisak ne bi „isključio“ istinu.
   - Prazan pritisak smanjuje verovatnoću ispod jastučeta.
4. **Sledeći pritisak** ide tamo gde se očekuje najveći dobitak informacije. Posle praznog pritiska senzor se vraća
   pored poslednjeg pritiska koji je dodirnuo viljušku.
5. **Kraj:** kad 90 % oblast padne ispod 4 mm² ili posle 10 pritisaka.

### 7.2 Glavna slika prezentacije: drška

- **Posle 1. pritiska na dršci** (`slike/07_drska_posle_1_pritiska.png`):
  - verovanje je **dugačka traka duž drške**, 90 % oblast je 218 mm²;
  - procena je 16 mm od težišta, a strana je još pogrešna (58 %).
  - Mreža „zna da ne zna“: pravo težište je unutar trake.
- **Posle 2. pritiska** (`slike/08_drska_posle_2_pritiska.png`): oblast je 18 mm², greška 0,84 mm.

### 7.3 Šta je na slici

- **Zelena providna viljuška** je procenjena poza: težište, smer i strana.
- **Zelena kugla** je procena, **crvena kugla** pravo težište.
- **Narandžasto** je 90 % oblast, a **plavi diskovi** su dosadašnji pritisci.
- **Dole levo:** taktilna slika i mapa verovatnoće.
- **Gore desno (samo u prozoru):** pogled odozgo na pravu viljušku (sivo) i viljušku postavljenu po proceni
  (zeleno: procenjeno težište, ugao i strana), sa tačkama pritisaka i poslednjim jastučićem.
  - Ispod je tabela: procena i istina za x, y (mm), ugao (°) i stranu, zatim greška težišta i ugla, 90 % oblast,
    broj pritisaka i deo pod poslednjim pritiskom. Pogrešna strana je crvena.
- **Senka ne ulazi u simulaciju senzora.** Taktilna kamera je ne crta, što je provereno bit po bit.

---

## 8. Rezultati (3 min)

### 8.1 Jedan pritisak

Konačni kandidat, 1.000 test pritisaka, po 250 po delu, čist ulaz:

| deo | medijana greške | p90 | unutar 5 mm |
|---|---|---|---|
| zupci | 0,86 mm | 1,45 mm | 99 % |
| dlan | 0,83 mm | 1,26 mm | 98 % |
| vrat | 0,80 mm | 1,94 mm | 96 % |
| drška | 1,07 mm | **23,3 mm** | 82 % |
| sve | 0,87 mm | 2,14 mm | 94 % |

Veliki p90 na dršci je dvosmislenost duž drške iz odeljka 7. Razrešava se drugim pritiskom.

### 8.2 Pretraga

300 epizoda, isti počeci za sve modele:

| | kandidat (`mini_robust`) | prethodni (`mini_exact`, 3 seed-a) |
|---|---|---|
| 90 % oblast < 16 mm² | 100 % epizoda, medijana 2 pritiska | 100 %, medijana 2 pritiska |
| < 16 mm² za najviše 2 / 3 pritiska | 80 % / 97 % | 95 % / 99,7 % |
| konačna greška, medijana / p90 | **0,62 / 1,18 mm** | 0,71 / 1,26 mm |
| greška > 5 mm | 0 % | 0 % |
| pravo težište u 90 % oblasti | **95 %** | 83 % |
| rano zaustavljanje (< 4 mm²) | 49 % | 69 % |

- **Po delu prvog pritiska**, konačna medijana: zupci 0,60, dlan 0,65, vrat 0,65, drška 0,63 mm. Početak na
  dršci više nije problem.
- **Smer i strana na kraju pretrage:**
  - greška smera: medijana 0,4°, p90 1,4°;
  - smer okrenut za 180° u 4 % epizoda;
  - strana tačna u 90,7 %.
- **Vreme:** u simulaciji oko 0,7 s po pritisku, sa fizikom i renderovanjem.

### 8.3 Poređenje sa metodom bez učenja (CAD šabloni)

- **Metod:** rekonstruisana dubina se poredi sa šablonima iz CAD-a za svaki položaj i ugao; prvo grubo, pa fino.
  Treninga nema.
- **Jedan pritisak:**
  - medijana 4,96 mm, a mreža 0,87 mm;
  - unutar 5 mm: 50 %, a mreža 94 %.
- **Pretraga:**
  - konačna greška 4,09 mm, a mreža 0,62 mm;
  - CAD završava dalje od 5 mm u 41 % epizoda, mreža u 0 %;
  - CAD uvek troši svih 10 pritisaka.
- **Kalibracija verovatnoća** (`slike/10_kalibracija_mreza_vs_cad.png`):
  - mreža je malo oprezna, sa krivom iznad dijagonale;
  - CAD je na zupcima preterano siguran;
  - CAD-ova 90 % oblast ima oko 12.000 mm², tj. skoro ništa ne kaže.
- **Pošteno:** CAD metod je jednostavna sopstvena implementacija, donja granica za metode bez učenja, a ne jak
  konkurent. Izgleda i da njegovi šabloni nisu u istoj razmeri kao dubina; to nije istraživano.

**Pokaži:** `slike/09_mreza_vs_cad_jedan_pritisak.png`.

---

## 9. Šta smo naučili: na šta je mreža zaista osetljiva (4 min, deo za profesore)

Ovo je verovatno najzanimljiviji deo za akademsku publiku.

### 9.1 Mreža određuje mesto na dršci iz širine drške

- **Problem:** drška se sužava linearno, širina je 4,0 − 0,05·x mm (8,0 mm na kraju, 5,2 mm kod težišta). Na
  komadu od 18 mm to je jedini lokalni znak gde je senzor duž drške.
- **Test** (509 pritisaka sa sredine drške): slika se raširi ili suzi poprečno na osu drške za faktor s i meri se
  pomak procene duž drške.
- **Očekivanje:** ako mreža čita širinu, pomak je (s − 1) · širina / 0,05.

| s | pomak procene | očekivano iz geometrije |
|---|---|---|
| 0,90 | −12,2 mm | −13,0 mm |
| 0,95 | −6,3 mm | −6,5 mm |
| 1,05 | +5,6 mm | +6,5 mm |
| 1,10 | +10,8 mm | +13,0 mm |

- Bočni pomak je ispod 0,06 mm.
- **Zaključak:** mreža meri apsolutnu širinu drške sa tačnošću ispod piksela i iz nje čita mesto. U simulaciji
  je to ispravan znak, jer je viljuška uvek ista i savršena.
- **Posledice:**
  - druga viljuška, bez ovog suženja, ne bi radila bez novog treninga;
  - svaka greška razmere mm/px pomera procenu.

### 9.2 Osetljivost na obradu slike: pronađena i ispravljena

- **Šta smo našli:** prethodni model je u pretrazi zavisio od sitnica obrade slike.
  - Kad je samo spoljni red piksela potamnjen, kao posle simuliranog skaliranja, procena se pomeri duž drške za
    +0,64 mm.
  - Raspon preko 8 načina obrade bio je 1,8 mm.
  - Mreža je kao znak koristila i **artefakt na ivici slike**.
- **Ispravka u treningu** (`mini_robust`):
  - nasumično prevzorkovanje (4 vrste interpolacije, pomak 0–1 px, zum 0,97–1,03; 25 % slika netaknuto);
  - nasumičan tretman svake ivice posebno;
  - na testu je spoljni pojas od 2 px uvek na nuli.
- **Rezultat:**
  - svi načini obrade na razmeri 1 daju istu pretragu, 0,62–0,65 mm;
  - ivica više ništa ne menja.
- **Šta nije uspelo:**
  - usrednjavanje na testu (11 varijanti slike) ne menja ništa, a košta 12–17 puta više vremena; odbačeno;
  - u toku je poslednji dozvoljeni pokušaj (zamućenje ulaza, σ = 1 px).
- **Preostala razlika nije artefakt nego prava razmera** (9.3).

### 9.3 Razmera mm/px: glavni zahtev za pravi senzor

- **U pretrazi:** 1 % greške razmere pomera procenu duž drške za oko 0,74 mm. Pri razmeri 0,98 greška raste sa
  0,62 na 1,78 mm, a težište je u 90 % oblasti samo u 21 % epizoda.
- **Kad se taj pomak oduzme,** greška je 0,56–0,66 mm za svaki ulaz; sva razlika je, dakle, taj pomak.
- **Zahtev:** mm/px mora biti poznat na oko **±0,3 %**.
- **Kalibracija u simulaciji:**
  - etalon sa dve kuglice: 95 % ponavljanja ≤ 0,07 %, dovoljno;
  - pomeraji robota: ≤ 0,27 %, na granici.

### 9.4 Drift osvetljenja je glavni izvor šuma

- **Lažna visina u rekonstruisanoj dubini**, medijana najveće po pritisku:
  - drift 3 %: 0,11 mm;
  - drift 10 %: 0,58 mm;
  - tekstura gela: 0,01–0,02 mm.
- **Mreža** je trenirana sa driftom: na driftu 10 % medijana je 0,89 mm, a na čistoj slici 0,87 mm.
- **CAD metod** na zupcima pod driftom pada sa 6,8 na 25,8 mm.
- **Za pravi senzor:** referentni kadar treba često obnavljati.

**Pokaži po želji:** `slike/11_mini_vs_standardni_senzor.png`. Manje polje Mini-ja košta skoro samo na dršci.

---

## 10. Istorija odluka u jednom pasusu (1 min)

- **v1:** mreža za težište na standardnom senzoru.
- **v2:** augmentacija šumom i driftom (nestali promašaji na podignutim zupcima) i pretraga sa više pritisaka.
- **v3:** više podataka se isplati, ali sve manje (2k → 40k: p90 9,6 → 2,1 mm). Tri seed-a se slažu na ±0,005 mm.
  Mreža je mnogo bolja od CAD metoda.
- **Mini:** polje pravog senzora, 100k podataka.
- **mini_exact:** ispravka greške u postupku, jer mreža nikad nije videla neprevzorkovanu sliku.
- **mini_robust:** nezavisnost od obrade slike i ivice.

Svaki korak je pokrenut merenjem koje je pokazalo konkretan problem.

---

## 11. Ograničenja (1 min, reći otvoreno)

- **Samo simulacija.** Pravi senzor i robot još nisu korišćeni.
- **Jedna viljuška, pričvršćena.** Model koristi tačno suženje ove drške.
- **Razmera:** greška mm/px od 1 % pomera rezultat za oko 0,7 mm. Kriterijum potpune nezavisnosti od obrade slike
  nije ispunjen.
- **Pretpostavke bez merenja:**
  - gel pravog Mini-ja (debljina 4,25 mm, mekoća) nije modelovan;
  - modeli šuma su pretpostavke, a ne merenja.
- **Jedan seed:** kandidat je treniran jednom. Kod prethodnog modela rasipanje p90 između seed-ova je 0,07 mm.
- **Najgora epizoda** (`slike/12_najgora_epizoda.png`, `video/04_najgora_start837.mp4`): 2,18 mm i pogrešna
  strana.

---

## 12. Sledeći koraci: pravi senzor (1 min)

1. **Kalibracija mm/px** na ±0,3 % etalonom sa dve kuglice (`calibrate_real.py`). Najvažniji pojedinačni zahtev.
2. **Ista obrada slike** kao u simulatoru:
   - referentni kadar;
   - fotometrijska kalibracija loptom;
   - rekonstrukcija istom funkcijom;
   - smanjenje na 0,2 mm/px sa INTER_AREA.
3. **Merenje šuma**, naročito drifta (`measure_noise.py`); augmentaciju podesiti na izmerene vrednosti.
4. **Provera suženja** prave viljuške u odnosu na CAD.
5. **Prvi test na hardveru:** nekoliko pritisaka na poznatim mestima, da se vidi da li se procena pomera duž drške
   (znak greške razmere).

Skripte za snimanje, kalibraciju i merenje šuma su napisane i proverene na simuliranim snimcima, ali još nisu
pokrenute na hardveru.

---

## 13. Demo scenario (5–10 min)

### Demo 1: ručni pritisak (`bash prezentacija/demo.sh 1`), 2 min

1. U desnom panelu otvori **Control**.
2. Klizač **press** na 3 → senzor se spušta na zupce.
   - Dole levo je taktilna slika: „FORK p=0.95, 4 tines“.
   - **Reci:** „Ovo je slika koju bi pravi senzor dao: boja kodira nagib, tačke su markeri.“
3. Klizač **x** na −0,03 → senzor klizi na dršku: „not a fork“.
   - **Reci:** „Sama drška je traka. Klasični detektor tu nema šta da kaže, i zato je bila potrebna mreža.“
4. **press** na −2 → senzor se podiže. **Page Down** menja predmet (druge viljuške, kašika, češalj...), **Home** vraća na
   početak.

Ovaj demo koristi standardno polje (32 × 24 mm). Mreža i pretraga koriste polje Mini-ja.

### Demo 2: pretraga, početak na dršci (`bash prezentacija/demo.sh 2`), 3 min

1. **F6** (jedan pritisak) → pritisak na dršci.
   - Dole levo je mapa: duga traka duž drške, oblast 218 mm².
   - Gore desno je zelena viljuška pomerena 16 mm duž drške, sa pogrešnom stranom („up 58 %“, crveno). Ugao je
     tačan (greška 0,1°).
   - **Reci:** „Iz jednog otiska drške ne zna se gde je duž drške, i mreža to iskreno kaže.“
2. **F6** ponovo → oblast 18 mm², greška 0,84 mm, strana tačna; zelena i siva viljuška se poklapaju.
   - **Reci:** „Drugi pritisak, na mestu koje najviše razrešava, preseca traku.“
3. **F7** → pretraga ide do kraja (prema snimku: 10 pritisaka, 1,39 mm).
   - **Reci:** „Medijana preko 300 epizoda je 0,62 mm.“
4. **T** (providna scena) da se vidi zelena senka ispod senzora.

### Demo 3: nasumična viljuška, 2 min

- **F9** → nova nasumična poza viljuške, pa **F7**. Može se ponoviti više puta.
- Pre-snimljeni počeci, ako je potrebno: `demo.sh 3` (zupci, licem nadole) i `demo.sh 4` (vrat, licem nagore).

### Rezerva ako prozor ne radi

`video/01_drska_start25.mp4`, `02_zupci_start18.mp4`, `03_licem_nagore_vrat_start10.mp4` i
`04_najgora_start837.mp4`. To je isti kod, snimljen bez prozora.

---

## 14. Verovatna pitanja i kratki odgovori

**Zašto simulacija, a ne odmah pravi senzor?**
- Oznake (tačno težište, deo, strana) su besplatne i tačne.
- 100.000 pritisaka nije realno snimiti.
- Poremećaji (šum, razmera, obrada slike) mogu se uvoditi kontrolisano, jedan po jedan.
- Pravi senzor je sledeći korak, a alati za njega su spremni.

**Koliko je simulacija gela realna?**
- To je MuJoCo meki kontakt sa jednostavnim modelom deformacije, a ne metod konačnih elemenata (FEM).
- Fotometrija je Lambertova, a parametri nisu izmereni na pravom Mini-ju.
- Upravo zato mreža vidi rekonstruisanu **geometriju** (dubinu), a ne sirovu sliku u boji: manje zavisi od
  optike. Trening uključuje šum i drift.
- Ovo je najveći otvoreni rizik prenosa.

**Zašto ne regresija (x, y)?**
Dvosmislenost na dršci: regresija daje prosek, a mapa daje traku ili dva vrha. Mape se množe preko pritisaka,
a verovatnoće su proverljive. Detaljno u odeljku 6.2.

**Kako znate da su verovatnoće tačne?**
- Kalibracione krive (`slike/10_...`): kod mreže su malo iznad dijagonale, tj. mreža je oprezna.
- Na kraju pretrage pravo težište je u 90 % oblasti u 95 % epizoda.

**Da li radi za drugu viljušku?**
- **Ne bez novog treninga.** Test suženja pokazuje da mreža koristi tačnu širinu ove drške.
- **Moguća rešenja:**
  - trening na porodici oblika (više CAD viljuški);
  - pretraga koja namerno ide ka vratu i zupcima, gde su znaci oblika jaki.

**Zašto ne klasično poklapanje sa CAD-om?**
Isprobano je na istim pritiscima: 4,96 prema 0,87 mm za jedan pritisak, 41 % prema 0 % promašaja u pretrazi.
Ravna drška daje skoro isto poklapanje na mnogo mesta.

**Kako se bira sledeći pritisak?**
Po najvećem očekivanom dobitku informacije, pod jednostavnim modelom: jastučić ili pokrije težište, ili ostaje
ostatak raspodele. Posle praznog pritiska senzor se vraća uz poslednji kontakt.

**Šta ako se viljuška pomeri pri pritisku?**
- U ovom radu je pričvršćena.
- Podaci beleže pritiske pri kojima bi se slobodna viljuška pomerila (`would_move`), a postoji i verzija
  simulatora sa slobodnim telom. Trening sa pomeranjem nije rađen.

**Koliko traje?**
- Trening: oko 4,5 h na GPU.
- Zaključivanje: milisekunde.
- Pretraga do 90 % oblasti ispod 16 mm²: medijana 2 pritiska.

**Koliko je rezultat stabilan?**
- Prethodni model, 3 seed-a: p90 1,84 ± 0,07 mm, konačna greška pretrage 0,71 ± 0,04 mm.
- Kandidat je treniran samo jednom. To je navedeno kao ograničenje.

**Šta je najveći rizik za pravi senzor?**
Kalibracija mm/px (±0,3 %), drift osvetljenja i razlika između pravog i simuliranog gela.

---

## 15. Izvori brojeva

Putanje su relativne prema korenu projekta.

| šta | fajl |
|---|---|
| fizika, optika, senzor, CAD provere | `README.md` (odeljci „The moving-sensor setup“, „Objects from STL files“), `gelsight_sim.py` |
| greška rekonstrukcije dubine | `README.md`, „Reconstructed vs ground-truth depth“ |
| klasični detektor | `README.md`, „Detector results“; `REZIME.md`, odeljak 3 |
| skup podataka | `dataset_mini/metadata.json`, `dataset_mini/cache_mini.npz` |
| arhitektura, trening | `train_centre.py` (`Net`), `centre_v3.py` (`train_size`), `results_centre/mini_robust/model_100k_seed0.json` |
| jedan pritisak po delu, CAD, drift | `results_centre/mini_cad/compare_cad_vs_net_mini_robust_s0.json` |
| pretraga (kandidat) | `results_centre/mini_robust/search/search_metrics.json` |
| pretraga (mini_exact, 3 seed-a) | `results_centre/mini_exact/seeds_100k.json`, `results_centre/mini_robust/diag/criterion.json` |
| pretraga CAD | `results_centre/mini_cad/search/search_metrics.json` |
| test suženja drške | `results_centre/taper_test/taper_mini_robust_s0.json` |
| razmera, obrada slike | `results_centre/mini_robust/diag/criterion.json`, `results_centre/mini_robust/input_scale_sweep_wide.json` |
| TTA | `results_centre/mini_robust_tta/diag/criterion.json`, `results_centre/mini_robust_tta/runtime_tta.json` |
| kalibracija mm/px | `results_centre/v2/calibration.json` |
| drift, tekstura | `results_centre/noise_effect/recon_noise_effect.json` |
| smer i strana | `results_centre/viz/mini_robust_s0/yaw_search.json`, `yaw_heuristic.json` |
| videi, kadrovi | `results_centre/videos/mini_robust_s0/index.json` |
| vreme zaključivanja (standardni senzor) | `results_centre/v3/runtime.json` |
| istorija v1–v3, Mini | `REZIME.md` |
