# Izvještaj o provjeri — VisA YOLO tools 1.0.0

Datum: 21. septembar 2026.

## Rezultat

**38/38 automatskih regresijskih testova: PASS.**

Komanda, iz foldera `visa_yolo_tools`:

```bash
python -m unittest discover -s tests -v
```

Izvršenje je provjereno na Linuxu uz Python 3.13.5, NumPy 2.3.5, OpenCV 4.13.0 i Pillow 12.3.0. Podrška za Python 3.10+ zasniva se na korištenom jezičkom/API skupu i deklarisanim zavisnostima; nije zasebno izvršena matrica testova svih Python verzija. PowerShell upute nisu izvršene na Windowsu.

## Šta pokriva regresijska provjera

- Maske s malim indeksima 1–6, binarne 0/255 maske, 16-bitni indeksi i paletni PNG gdje je boja defekta crna, a pozadina bijela. Paletni indeksi se ne pretvaraju u svjetlinu.
- Odvojene komponente, pun okvir slike, jedan piksel na donjoj desnoj granici, dijagonalna povezanost 4/8, spajanje u union okvir, padding i njegovo ograničenje na dimenzije slike.
- YOLO koordinate i normalizacija, odbacivanje nevažećih okvira, filtriranje prema broju maskiranih piksela i uklanjanje identičnih izlaznih okvira.
- Prihvatljiv grayscale RGB zapis i odbijanje proizvoljnih colorized RGB/transparentnih maski.
- Cjelovita konverzija sintetičke strukture originalnog VisA rasporeda, generisanje YAML/CSV/JSONL fajlova, preview slika i HTML pregleda.
- Ispravne slike sa praznim anotacijama, izvorna imena koja su ista u Normal i Anomaly, kopije bajt-po-bajt identične originalima i očuvanje izvora.
- Ponovljivost splita i anotacija, planiranje bez pisanja (`--check-only`), tačni brojevi za 503/100 uzoraka i minimalna veličina skupova.
- Nedostajuće, prazne, višeznačne i orphan maske; različite dimenzije; korumpirane slike i EXIF rotacija.
- Identifikacija potpuno identičnih RGB piksela, zadržavanje duplikata u jednom splitu i odbijanje konfliktnih anotacija istih slika.
- Zaštita postojećeg izlaza i izvornog Data foldera; uklanjanje samo novog privremenog izlaza poslije simulirane greške pisanja.

## Dodatne izvršene provjere

**CLI smoke test: PASS.** Skripta je pokrenuta kao samostalan program nad **603 sintetičke slike**, raspoređene kao 503 normalne i 100 defektnih. Izlaz je imao 422 train, 91 val i 90 test slika, s ispravnim parovima image/label. Svaka sintetička defektna slika imala je dva namjerno nacrtana područja, pa je ovaj test dao 200 okvira. **To nije broj okvira stvarnog VisA dataseta.**

**Nezavisna geometrijska provjera: PASS.** Izvršen je dodatni jednokratni audit 300 slučajnih maski za obje povezanosti, ukupno 600 slučajeva. Rezultati su upoređeni s nezavisnim Python flood-fill algoritmom i provjerena je povratna konverzija normalizovanih YOLO koordinata u piksele. Ovaj dodatni audit nije dio gore navedene komande za 38 regresijskih testova.

**Syntax/bytecode provjera: PASS.** Izvršen je `python -m compileall` nad kodom.

## Granice provjere

**Puna originalna VisA arhiva nije preuzeta ni obrađena u okviru ove provjere.** Struktura originalnih foldera, pravilo uparivanja imena i značenje nenultih maski provjereni su u službenom Amazonovom repozitoriju; svi izvršni testovi koristili su sintetičke podatke.

**Nije instaliran/pokrenut Ultralytics trening ni inference.** Izlazni fajlovi provjereni su u odnosu na dokumentovani YOLO detection format, bez izvođenja stvarnog treninga.

Provjera ne mjeri kvalitet naučenog modela, semantički optimalno grupisanje pukotina u okvire, near-duplicate leakage ili korisnost eksperimenta u proizvodnji. Prije treninga pregledaj generisani `previews/index.html` i `conversion_report.json` na svom stvarnom datasetu.
