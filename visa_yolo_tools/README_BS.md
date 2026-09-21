# VisA chewinggum → YOLO bounding box anotacije

Lokalna Python skripta za pretvaranje postojećih VisA maski u **Ultralytics YOLO detection** dataset. Zadana kategorija je `chewinggum`, a izlaz ima jednu klasu: **`0: defect`**.

Skripta ne trenira model, ne pokreće inference, ne preuzima slike i ne zahtijeva GPU, PyTorch, CUDA, AWS račun ili Ultralytics instalaciju. Nakon instalacije zavisnosti radi lokalno, bez mrežnih zahtjeva. Originalni dataset se ne mijenja.

**U ovom repozitoriju nema VisA slika niti maski.** Potrebna je raspakovana originalna VisA arhiva, koja se preuzima zasebno [1]. `data.yaml` se generiše nakon konverzije, kada je poznata stvarna izlazna putanja.

## 1. Raspored foldera

Kloniraj ovaj repozitorij ili kopiraj samo folder `visa_yolo_tools`, tako da se `visa_yolo_tools` nalazi uz folder `chewinggum`. Ako si klonirao repozitorij, VisA kategorije možeš raspakovati direktno u njegov korijen: isključene su u `.gitignore`.

```text
VisA/
├── chewinggum/
│   └── Data/
│       ├── Images/
│       │   ├── Normal/
│       │   │   ├── 000.JPG
│       │   │   └── ...
│       │   └── Anomaly/
│       │       ├── 000.JPG
│       │       └── ...
│       └── Masks/
│           └── Anomaly/
│               ├── 000.png
│               └── ...
├── visa_yolo_tools/
│   ├── convert_visa_to_yolo.py
│   ├── requirements.txt
│   ├── README_BS.md
│   ├── README_EN.md
│   ├── TEST_REPORT_BS.md
│   ├── TEST_REPORT_EN.md
│   ├── SHA256SUMS.txt
│   ├── SOURCE_ATTRIBUTION.md
│   └── tests/
│       └── test_converter.py
└── ... ostale kategorije mogu ostati ovdje
```

Primjeri naziva su ilustrativni. Bitno je da se slika i njena maska unutar `Anomaly` podudaraju po nazivu bez ekstenzije: npr. `043.JPG` i `043.png`. Nazivi u `Normal` i `Anomaly` mogu biti isti: izlaz dodaje različite prefikse da ne bi došlo do prepisivanja.

Skripta čita **originalni** raspored `Data/Images/...` i `Data/Masks/...` potvrđen u Amazonovom repozitoriju [1, 3]. Ne koristi `image_anno.csv` ni originalne `split_csv` podjele: novo razdvajanje se radi iz svih slika izabrane kategorije. Ne podržava reorganizovani `VisA_pytorch/1cls` raspored, ZIP/TAR arhivu koja nije raspakovana, niti proizvoljne RGB vizualizacije maski.

## 2. Pokretanje — Linux / WSL

Potrebni su Python **3.10 ili noviji**, `venv` i `pip`. Otvori terminal u svom `VisA` folderu:

```bash
cd "/putanja/do/VisA"

python3 -m venv .venv-visa
source .venv-visa/bin/activate
python -m pip install -r visa_yolo_tools/requirements.txt

python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum
```

Time dobiješ novu lokalnu kopiju dataseta u `VisA/yolo_chewinggum/`. Potreban je dodatni prostor za kopije slika odabrane kategorije i pregledne slike; ostale VisA kategorije se ne kopiraju.

Pri narednom korištenju istog virtualnog okruženja samo aktiviraj `.venv-visa`; ne moraš ga ponovo kreirati ni instalirati pakete.

### Windows PowerShell

Iz `VisA` foldera:

```powershell
py -3 -m venv .venv-visa
.\.venv-visa\Scripts\python.exe -m pip install -r .\visa_yolo_tools\requirements.txt
.\.venv-visa\Scripts\python.exe .\visa_yolo_tools\convert_visa_to_yolo.py --source . --output .\yolo_chewinggum
```

Aktivacija PowerShell okruženja nije potrebna, pa nije potrebno mijenjati execution policy. `py -3` mora pokazivati na Python 3.10 ili noviji. PowerShell komande su navedene kao upute; skripta je testirana na Linuxu.

### Druga lokacija izvora

`--source` može pokazivati na `VisA`, njegov direktni nadređeni folder ili direktno na `chewinggum`:

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source "/mnt/datasets/VisA/chewinggum" \
  --output "/mnt/datasets/yolo_chewinggum"
```

Izlaz je relativan trenutnom radnom folderu ako nije naveden apsolutnom putanjom. Skripta obrađuje samo jednu kategoriju po pokretanju.

## 3. Šta je zadano

| Postavka | Vrijednost |
|---|---|
| Kategorija | `chewinggum` |
| Klase | Jedna: `0 = defect` |
| Foreground maske | Svi pikseli čija je vrijednost veća od nule |
| Okviri | Jedan po povezanom području foregrounda |
| Povezanost | 8 susjeda, uključujući dijagonale |
| Minimalna površina | 1 piksel — ništa se ne odbacuje po veličini |
| Dodatna margina | 0 piksela |
| Train / validation / test | Približno 70% / 15% / 15%, odvojeno za normalne i defektne slike |
| Seed | `42` |
| Vizuelni pregled | Do 24 primjera iz train/val, s maskama i okvirima |
| Slike za treniranje | Originalne slike kopirane bez promjene bajtova |
| Normalne slike | Prazni `.txt` fajlovi, bez klase `good` |

Za kompletan objavljeni `chewinggum` skup od 503 ispravne i 100 defektnih slika [1], **ako nema grupa identičnih slika**, zadana podjela daje:

| Skup | Ispravne | Defektne | Ukupno |
|---|---:|---:|---:|
| train | 352 | 70 | 422 |
| val | 76 | 15 | 91 |
| test | 75 | 15 | 90 |
| Ukupno | 503 | 100 | 603 |

Broj bounding boxova nije jednak broju defektnih slika: jedna slika može imati više okvira. Tačan broj dobiješ nakon obrade svojih maski.

Skripta može raditi i na namjerno odabranom manjem skupu. Potrebne su najmanje tri različite grupe slika za svaki od dva uslova, normal/anomaly, da svaki izlazni skup ima oba uslova. Kod vrlo malih skupova omjeri se prilagođavaju tom minimumu.

## 4. Izlazni fajlovi

```text
yolo_chewinggum/
├── data.yaml
├── classes.txt
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── previews/
│   ├── index.html
│   └── ... .jpg slike za pregled
├── manifest.csv
├── annotations.jsonl
├── conversion_report.json
└── SOURCE_ATTRIBUTION.md
```

`data.yaml` definiše klasu i foldere u Ultralytics formatu [4]. Njegovo polje `path` sadrži **apsolutnu putanju izlaznog foldera**. Nakon prenosa generisanog dataseta na drugi računar/server promijeni `path` na novu apsolutnu putanju; train/val/test ostaju relativni toj putanji. YAML putanju s Windows diskom piši uz `/`, npr. `D:/datasets/yolo_chewinggum`.

`manifest.csv` povezuje svaku izvornu sliku i masku s izlaznim fajlovima, uslovom i splitom, dimenzijama, brojem okvira, originalnim vrijednostima maske i SHA-256 hashovima. Putanje `source_image` i `source_mask` relativne su izvornom folderu kategorije (`chewinggum`), ne vrhu `VisA` arhive.

`annotations.jsonl` dodatno čuva piksel-koordinate okvira radi provjere. `conversion_report.json` čuva parametre, seed, statistiku, upozorenja, verzije biblioteka, hash skripte i grupe identičnih slika.

### Vizuelna provjera

Otvori **`yolo_chewinggum/previews/index.html`** u browseru, bez pokretanja servera. Klikom na sliku otvaraš njen pregled u punoj rezoluciji.

Crveni poluprozirni sloj je postojeća maska. Zeleni pravougaonici su izračunati bounding boxovi. To su **anotacije, a ne predikcije modela**. Ispravni primjerci nemaju ni masku ni okvire.

Pregledne slike se nikada ne ubacuju u `images/` i ne smiju se koristiti kao ulaz za trening. Test slike se namjerno ne prikazuju u automatskom pregledu: za pregled pravila konverzije koristi train/val, a test ostavi za završnu evaluaciju.

## 5. Dodatne opcije

### Provjera izvora bez kreiranja dataseta

```bash
python visa_yolo_tools/convert_visa_to_yolo.py --source . --check-only
```

Ovo učitava i provjerava slike/maske, računa okvire i planira podjelu, ali ne piše izlazni dataset.

### Druga podjela ili više preglednih slika

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum_80_10_10 \
  --split 0.80 0.10 0.10 \
  --seed 42 \
  --previews 48
```

Sva tri omjera moraju biti pozitivna i njihov zbir mora biti 1. Ponavljanje istog skupa, verzije skripte, parametara i seeda daje istu podjelu. Nakon dodavanja ili uklanjanja izvornih slika može se promijeniti i podjela ostalih slika; za ponovljiv eksperiment sačuvaj generisani dataset i manifest.

### Jedan okvir oko svih defekata na slici

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum_union \
  --box-mode union
```

`union` je opcionalan: može spojiti udaljena oštećenja u vrlo širok pravougaonik koji obuhvata i ispravne dijelove. Zadani `components` je precizniji u smislu lokalizacije odvojenih područja, ali fragmentirane maske mogu proizvesti mnogo okvira.

### Margina ili filtriranje sitnih komponenti

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --output ./yolo_chewinggum_padded \
  --padding 2
```

`--min-area 5` odbacuje povezane komponente sa manje od pet označenih piksela **u originalnoj rezoluciji maske**. Mjera je broj foreground piksela, ne površina pravougaonika. Ovo nije zadano: legitimne sitne pukotine mogu nestati. Svako odbacivanje se evidentira. Ako bi neka defektna slika izgubila sve okvire, skripta prekida rad umjesto da je pogrešno proglasi ispravnom.

`--connectivity 4` ne smatra dijagonalno dodirivanje povezanošću. Za ovaj primjer ostavi zadanu vrijednost 8.

`--previews 0` isključuje pregledne slike. Sve opcije vidiš ovako:

```bash
python visa_yolo_tools/convert_visa_to_yolo.py --help
```

### Druga originalna VisA kategorija

```bash
python visa_yolo_tools/convert_visa_to_yolo.py \
  --source . \
  --category cashew \
  --output ./yolo_cashew
```

I ovdje svi nenulti tipovi defekata postaju ista klasa `defect`. Skripta ne generiše zasebne YOLO klase za pojedinačne tipove oštećenja.

## 6. Zaštite i ograničenja

**Maske.** Čitaju se sirove indeksne vrijednosti, uključujući paletni PNG (`P` mode), bez pretvaranja palete u grayscale. To je važno jer indeks 1 ne mora imati svjetlinu 1 [6]. Podržane su i binarne 0/255 maske unutar originalnog rasporeda. Proizvoljne colorized RGB maske se odbijaju; grayscale zapisan kroz identične RGB kanale je prihvatljiv. Kod RGBA/LA maski neprozirna alpha je obavezna.

**Poklapanje.** Nedostajuća maska, maska bez slike, višeznačno ime fajla, oštećena slika, prazna maska defektne slike i različite dimenzije slike i maske zaustavljaju obradu. Nema tihog preskakanja uzoraka i nema automatskog resizea. Slike/maske s EXIF rotacijom se odbijaju da koordinatni sistemi ne bi bili pogrešno poravnati.

**Piksel-koordinate.** Okvir obuhvata pune označene piksele. Zbog toga i komponenta od jednog piksela ima pozitivnu širinu i visinu. YOLO zapis je `class_id x_center y_center width height`, normalizovan na dimenzije slike [4]. Algoritam koristi OpenCV connected components [5]. Dodirnute klase iz maske mogu završiti u istom okviru jer su namjerno spojene u jednu foreground klasu.

**Duplikati i leakage.** Slike s potpuno identičnim dekodiranim RGB pikselima uvijek ostaju u istom splitu, uključujući fajlove koji se razlikuju samo u metapodacima. Ako iste slike imaju konfliktne anotacije, obrada se zaustavlja. Skripta **ne prepoznaje** slične slike, različite kadrove istog fizičkog proizvoda, istu seriju ili istu sesiju snimanja. Zbog toga ovo nije garancija odsustva svih oblika data leakagea. Podjela je stratificirana prema normal/anomaly statusu, **ne** prema svih šest tipova defekata.

**Pisanje.** Postojeći izlazni folder se nikada ne prepisuje niti briše. Za drugi pokušaj navedi novi `--output`. Izlaz ne može zamijeniti izvor ili njegove nadređene foldere niti biti unutar izvornog `Data/`. Obrada prvo validira ulaz, zatim piše u novi privremeni folder i tek nakon provjere objavljuje završni folder. U slučaju greške uklanja se samo privremeni folder koji je kreirala skripta. Kopije slika se provjeravaju pomoću SHA-256. Ne mijenjaj ulazne fajlove dok konverzija traje.

**Značenje klase.** Ovo je detekcija lokacije oštećenja, ne detekcija cijelog proizvoda, klasifikacija good/bad proizvoda ili mjerenje precizne granice defekta. Bounding box ne predstavlja nužno jedan fizički defekt: povezane komponente su geometrijska, ne semantička podjela. Odsustvo modelove detekcije kasnije nije dokaz da je proizvod ispravan.

## 7. Trening i licenca dataseta

Za trening u svom postojećem Ultralytics okruženju kao `data` koristi generisani **`yolo_chewinggum/data.yaml`**. Ovaj alat ne instalira niti pokreće trening.

Za odabir parametara koristi samo `train/val`, a `test` split ostavi za konačnu evaluaciju. Podjela je **novi nadzirani eksperiment**: rezultati na njoj nisu uporedivi s originalnim VisA anomaly-detection benchmarkom, čiji protokoli imaju drugačije podjele [1, 2].

VisA dataset je označen kao CC BY 4.0 u izvornom repozitoriju [1, 7] i ovaj alat ne mijenja tu licencu. Uz izvedeni dataset automatski se kopira `SOURCE_ATTRIBUTION.md`; ako objavljuješ ili dalje dijeliš slike ili izvedene podatke, sačuvaj atribuciju, referencu na licencu i opis izmjena.

## 8. Provjera skripte

Iz `VisA` foldera, nakon instalacije zavisnosti:

```bash
python -m unittest discover -s visa_yolo_tools/tests -v
```

Testovi kreiraju male sintetičke ulaze u sistemskom privremenom folderu, obrađuju ih i zatim ih uklanjaju. Ne diraju tvoj dataset. Detalji izvršene provjere i njena ograničenja nalaze se u `TEST_REPORT_BS.md`.

## Izvori

[1] Amazon Science — VisA opis, raspored foldera, broj uzoraka, licenca:
https://github.com/amazon-science/spot-diff

[2] Amazon Science — originalno čitanje/binarizacija maski i reorganizacija:
https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py

[3] Amazon Science — originalno uparivanje slika i maski; definicije indeksnih klasa:
https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv
https://github.com/amazon-science/spot-diff/blob/main/utils/id2class.py

[4] Ultralytics — YOLO detection format, normalizovane koordinate, dataset YAML:
https://docs.ultralytics.com/datasets/detect/

[5] OpenCV — connectedComponentsWithStats:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

[6] Pillow — image modes, palette/index pixels:
https://pillow.readthedocs.io/en/stable/handbook/concepts.html

[7] Dataset licenca:
https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET
https://creativecommons.org/licenses/by/4.0/

Izvorni raspored i format provjereni 21. septembra 2026. Ovo je zasebno napisana skripta za konverziju; nije službeni Amazonov niti Ultralyticsov alat.
