# VisA chewinggum → YOLO26n trening na SageMakeru

Alati za trening nalaze se u folderu **`training`** ovog repozitorija, uz konverter
u `visa_yolo_tools`. Sve komande pokreću se iz **roota repozitorija**.

Glavni tok je: lokalna priprema i validacija → S3 dataset/source kanali → jedan
SageMaker GPU training job → `model.tar.gz` sa `best.pt`, `last.pt` i metapodacima.
Ne moraš instalirati PyTorch, CUDA ili Ultralytics na laptop da bi pokrenuo ovaj tok.

Ovo je samostalan, demonstracijski primjer treninga. Nije produkcijski pipeline i
ne uključuje deployment modela.

## Brzi početak — Linux / WSL

Nijedna komanda ne pokreće AWS posao bez eksplicitnog `--execute`.

### 1. Raspored foldera

```text
VisA/
├── chewinggum/Data/Images/Normal/...
├── chewinggum/Data/Images/Anomaly/...
├── chewinggum/Data/Masks/Anomaly/...
├── yolo_chewinggum/                 # možda već postoji iz prethodnog koraka
├── visa_yolo_tools/                 # konverter maski u YOLO format
└── training/
    ├── run.py
    ├── config.yaml
    ├── config.local.example.yaml
    ├── requirements.txt             # samo lokalna priprema i AWS controller
    ├── source/                      # mali paket koji ide u SageMaker
    ├── tests/
    ├── weights/                     # yolo26n.pt ide ovdje
    └── work/                        # nastaje kasnije; nije u repozitoriju
```

Originalna VisA arhiva mora biti raspakovana. GitHub clone bez preuzetih slika nije
sam po sebi dataset. Originalne slike/maske nisu uključene u repozitorij.

Sve konfiguracijske relativne putanje računaju se od **roota repozitorija**, nezavisno
od trenutnog radnog foldera terminala. Ako ti je izvor npr. `datasets/VisA`, možeš
koristiti `prepare --source datasets/VisA`.

### 2. Instaliraj samo lokalne zavisnosti

Potreban je Python 3.10 ili noviji za lokalni controller. SageMaker image koristi
svoj odvojeni Python 3.10 i GPU biblioteke.

```bash
python3 -m venv training/.venv
source training/.venv/bin/activate
python -m pip install -r training/requirements.txt
```

Ne instaliraj stare zahtjeve originalnog `spot-diff` istraživačkog koda u ovo
okruženje. Ne treba pokretati njegov model ili originalni anomaly-detection trening.

### 3. Pripremi ili ponovo upotrijebi YOLO dataset

```bash
python training/run.py prepare
```

Ako `yolo_chewinggum` već postoji, provjerava se i koristi **ista podjela**, bez
novog dijeljenja ili prepisivanja. Ako ne postoji, konverter iz `visa_yolo_tools`
ga pravi iz originalnih slika i maski.

Konverzija koristi sve nenulte piksele maske, jednu klasu `0: defect`, povezane
komponente i približno 70/15/15 train/val/test, seed 42. Ispravne slike imaju prazne
`.txt` anotacije. Postojeće slike se ne precrtavaju. `previews/index.html` je samo
vizuelna provjera anotacija i njegove slike se ne koriste za trening.

Za novu podjelu ili drugačiju konverziju izaberi **novi izlazni dataset folder**;
postojeći dataset se namjerno ne prepisuje. Specijalne parametre konvertera vidi
u `python visa_yolo_tools/convert_visa_to_yolo.py --help`.

### 4. Dodaj pretrained weights

```bash
python training/run.py download-weights
```

Ova komanda preuzima samo zvanični Ultralytics release asset `yolo26n.pt` i čuva
SHA-256. Ne koristi PyTorch, ne pristupa AWS-u i ne pokreće trening.

Alternativa: kopiraj svoj pouzdani `yolo26n.pt` u `training/weights/`, ili promijeni
`paths.weights` na postojeći lokalni fajl. Postojeći weights fajl se ne prepisuje.
Ne učitavaj nepoznate/nepouzdane `.pt` fajlove; model checkpoint nije običan tekst.
Zabilježeni hash omogućava praćenje identiteta fajla, ali nije nezavisna provjera
potpisa izdavača. Weights nisu uključeni u repozitorij.

### 5. Unesi postojeći S3 bucket

```bash
cp training/config.local.example.yaml training/config.local.yaml
```

U `config.local.yaml` promijeni:

```yaml
aws:
  bucket: naziv-tvog-postojeceg-bucketa
```

Bucket mora biti u regiji iz `aws.region` (zadano **eu-central-1**), privatni i
dostupan tvojim AWS kredencijalima i SageMaker execution roli. Alat ne pravi bucket,
IAM role, endpoint ili deploy. Zadano se koristi standardni AWS credential chain
(npr. `AWS_PROFILE` ili default profil); imenovani profil i drugu regiju možeš
zadati kroz `aws.profile` i `aws.region` u `config.local.yaml`.

Zadano traži postojeću IAM rolu `SageMakerExecutionRole` u tvom AWS računu. Ako nemaš `iam:GetRole` ovlaštenje, unesi puni poznati ARN:

```yaml
aws:
  bucket: naziv-tvog-postojeceg-bucketa
  execution_role_arn: arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
```

Ne stavljaj AWS access key/secret/session token u fajlove. Koriste se tvoji postojeći
AWS credential provider i profil. Nijedan AWS account ID nije ugrađen u alat.

### 6. Pregledaj plan bez AWS poziva

```bash
python training/run.py plan --preset smoke
```

Provjerava dataset, sve tri podjele, moguće curenje identičnih slika, weights i
konfiguraciju te ispisuje zahtjev. **Nema AWS poziva, uploada niti troškova GPU-a.**
Ako je rola zadana samo imenom, plan prikazuje placeholder; ARN se razrješava tek
pri eksplicitnom submitu. Plan ne potvrđuje IAM ovlaštenja, postojanje GPU imagea,
kvotu ili dostupan kapacitet.

### 7. Pokreni mali plaćeni smoke trening

```bash
python training/run.py submit --preset smoke --execute
```

`--execute` je obavezan za S3 upload i kreiranje training joba. Bez njega `submit`
radi samo lokalni dry run. Prije uploada provjeravaju se AWS identitet, regija
bucketa i execution role. Ispisuju se ime joba, ARN i lokacija lokalne evidencije.

Smoke postavke: **1 epoha, batch 1, workers 0, imgsz 640, jedna ml.g4dn.xlarge**.
Ovo je provjera mehanike pipelinea, ne dokaz kvaliteta istreniranog modela.
Maksimalni runtime joba u ovom presetu je 1.800 sekundi. To nije procjena trajanja
ili ukupne cijene posla.

### 8. Puni trening

Nakon provjere smoke posla:

```bash
python training/run.py submit --preset full --execute
```

Ovaj preset koristi **60 epoha**, batch 1, workers 0 i runtime limit 14.400 sekundi.
Broj epoha je početna demonstracijska postavka, ne garantovana optimalna vrijednost.
Možeš eksplicitno promijeniti npr. batch ili rezoluciju:

```bash
python training/run.py plan --preset full --batch 4 --imgsz 960
python training/run.py submit --preset full --batch 4 --imgsz 960 --execute
```

Val skup se koristi za validaciju i izbor `best.pt`. **Test slike se zadano uopće
ne uploaduju.** Za završni eksperiment, kada su izbori i parametri zaključeni,
možeš jednom eksplicitno uključiti i evaluaciju na test skupu:

```bash
python training/run.py submit --preset full --evaluate-test --execute
```

Tada se test šalje u job, ali se koristi tek **nakon treninga**, za evaluaciju
odabranog `best.pt`. Nemoj naknadno prilagođavati parametre prema test rezultatima
pa ih predstavljati kao nepristrasnu finalnu evaluaciju.

## Praćenje, zaustavljanje i preuzimanje

Zamijeni `IME_JOBA` stvarnim imenom iz submit izlaza:

```bash
python training/run.py status --job-name IME_JOBA
python training/run.py status --job-name IME_JOBA --wait
python training/run.py download --job-name IME_JOBA
```

`download` radi kada je job `Completed`. Arhiva se provjerava prije raspakivanja:
nisu dozvoljene putanje izvan izlaza, simbolički linkovi ili prepisivanje postojećih
rezultata. Modeli i izvještaji završavaju ovdje:

```text
training/work/jobs/IME_JOBA/
├── create-training-job.json
├── config-snapshot.json
├── input-manifest.json
├── dataset-validation.json
├── submission.json
├── model.tar.gz
└── artifacts/
    ├── best.pt
    ├── last.pt
    ├── metrics.json
    ├── results.csv
    ├── environment.json
    ├── training_config.json
    ├── resolved_training_args.json
    ├── dataset_validation.json
    ├── data.yaml
    ├── classes.txt
    └── ... dostupni grafovi i izvorni metapodaci
```

`data.yaml` u model artefaktu je prenosiv opis, **ne sadrži dataset slike**. Za
lokalni inference/evaluaciju koristi stvarni `yolo_chewinggum/data.yaml` ili
postavi njegov `path` na lokaciju preuzetog dataseta.

CloudWatch logovi, preko AWS CLI-ja:

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "IME_JOBA/" \
  --follow --region eu-central-1
```

Zaustavljanje posla:

```bash
python training/run.py stop --job-name IME_JOBA --execute
```

**Zatvaranje terminala ili Ctrl+C na lokalnom čekanju NE zaustavlja SageMaker job.**
Stop komanda šalje zahtjev; statusom provjeri da je zaista `Stopped`. Alat ne
briše S3 ulaze, modele, logove ili job evidenciju. S3/log storage može ostati i
nakon završetka treninga. Nema automatskog ponavljanja posla pod drugim imenom.

Ako CreateTrainingJob poziv završi mrežnom greškom, prvo provjeri ispisano ime
joba kroz `status`/AWS konzolu: posao je možda kreiran iako odgovor nije stigao.

## Verzije i runtime postavke

| Stavka | Postavka |
|---|---|
| Task | `detect`, jedna klasa `defect` |
| Početni checkpoint | `yolo26n.pt`; bez zamjene drugim modelom |
| Ultralytics | `8.4.60`, namjerno zamrznuta referentna verzija |
| GPU container | AWS PyTorch 2.2.0 / Python 3.10 / CUDA 12.1 |
| AWS profil / regija | default credential chain / `eu-central-1` |
| Instance | 1 × `ml.g4dn.xlarge` |
| Seed | 42 |
| Optimizer | `auto`, odabir prepušten pinovanoj verziji Ultralytics |
| AMP | `false`, eksplicitni FP32 za ovaj primjer |
| Train/val/test | Nova nadzirana podjela, ne originalni VisA benchmark |

`amp=false` je izbor ovog primjera; time izbjegavamo pomoćnu AMP provjeru koja bi
mogla preuzeti dodatni model. Checkpoint se provjerava kao YOLO26 nano detection
prije treninga. Nedostupan GPU prekida izvršavanje umjesto tihog CPU fallbacka.

Bootstrap instalira pinovane Ultralytics/NumPy/OpenCV pakete **unutar GPU kontejnera**.
Constraints čuvaju PyTorch 2.2.0 i torchvision 0.17.0; konflikt prekida posao umjesto
skrivenog upgradea CUDA/PyTorch okruženja. Po potrebi instalira `libgl1` i `libglib2.0-0`
unutar kontejnera. Potreban je outbound pristup paketnim repozitorijima. Ovo nije
hermetički offline image niti security-auditiran produkcijski container; stvarno
instalirane verzije bilježe se u `environment.json`.

## Opcionalni ONNX export

```bash
python training/run.py submit --preset full --export-onnx --execute
```

Dodaje `best.onnx` i `onnx_metadata.json` nakon treninga. Export je FP32, batch 1,
static, opset 17, bez simplifikacije. Bilježe se stvarni nazivi i dimenzije I/O
umjesto izmišljanja izlaznog formata za neku verziju YOLO26.

Ovo je generički export, **nije potvrđen deployment ugovor za konkretan edge uređaj**.
Ne gradi TensorRT engine. ONNX export nije ovdje izvršen/testiran;
pokreni početni smoke sa `--export-onnx` prije punog posla kojem je ONNX obavezan.

## Opcionalni direktni lokalni trening

Glavni workflow iznad ostaje SageMaker. Za lokalni GPU koristi **odvojeno**
okruženje sa Python 3.10 ili 3.11, jer referentni PyTorch 2.2.0 nema wheel za novije
Python verzije. Primjer za Linux/WSL i kompatibilan NVIDIA/CUDA driver:

```bash
python3.10 -m venv training/.venv-gpu
training/.venv-gpu/bin/python -m pip install \
  torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
training/.venv-gpu/bin/python -m pip install \
  -c training/source/constraints.txt -r training/source/requirements.txt
python training/run.py local --preset smoke --python training/.venv-gpu/bin/python
```

`local` koristi isti `source/train.py`, ali lokalne izlazne putanje. Ne šalje ništa
u AWS. Za namjerni CPU test dodaj `--device cpu`; nema automatskog CPU fallbacka.
Ovaj lokalni GPU tok nije izvršen tokom provjere alata.

## Pristupi i očekivane greške

Lokalnom AWS profilu trebaju odgovarajući `sts:GetCallerIdentity`,
`s3:GetBucketLocation`, S3 upload/download, `sagemaker:CreateTrainingJob`,
`DescribeTrainingJob`, `StopTrainingJob`, `iam:PassRole`, te po potrebi `iam:GetRole`.
Za prikaz CloudWatch logova trebaju dodatna read ovlaštenja. Execution role mora
imati SageMaker trust policy i pristup odabranom S3 prefiksu, logovima i GPU imageu.
Uz KMS trebaju i odgovarajući key policy/grantovi. Ovo nije automatski IAM setup.

`ResourceLimitExceeded` obično zahtijeva provjeru SageMaker training quota za tačnu
instancu i regiju; EC2 quota nije ista stvar. Odobrena kvota nije garancija slobodnog
kapaciteta. Skripta ne mijenja kvote, region ili tip instance kao fallback.

Nijedan privatni repo, `.env`, kompletan repozitorij ili produkcijski dataset se ne
uploaduje. Upload je ograničen na konkretne validirane dataset fajlove, nekoliko
metapodataka, šest source fajlova, runtime konfiguraciju i izabrani checkpoint.

## Testovi, ograničenja i izvori

```bash
python -m unittest discover -s training/tests -v
```

Pogledaj `TEST_REPORT.md` za stvarno izvršene provjere. Offline testovi uključuju
mock/fake model radi provjere orchestration koda, **ne predstavljaju YOLO trening**.
Nije pokrenut AWS job, nije izmjerena preciznost modela, nisu preuzeti cijeli VisA
podaci ni weights, niti je izvršen ONNX export. Kao rezultate navodi isključivo
svoje stvarne trening/evaluacijske rezultate.

Izvori za javne tehničke ugovore (provjereni 21. septembra 2026):

- Ultralytics YOLO26, `yolo26n.pt`, train/val/export:
  https://docs.ultralytics.com/models/yolo26/
- Ultralytics training API:
  https://docs.ultralytics.com/modes/train/
- YOLO detection dataset format:
  https://docs.ultralytics.com/datasets/detect/
- Ultralytics package release history:
  https://pypi.org/project/ultralytics/
- Zvanični pretrained release:
  https://github.com/ultralytics/assets/releases/tag/v8.4.0
- AWS AlgorithmSpecification / ContainerEntrypoint:
  https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AlgorithmSpecification.html
- SageMaker File-mode kanali i njihove putanje:
  https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
- AWS PyTorch 2.2.0 container referenca:
  https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-data-parallel-support.html
- VisA dataset i anotacije:
  https://github.com/amazon-science/spot-diff

VisA podatke navedi uz CC BY 4.0 attribution i opis konverzije/podjele; pogledaj
`visa_yolo_tools/SOURCE_ATTRIBUTION.md`. Licenca dataseta nije zamjena za zasebne uslove korištenja
Ultralytics softvera/weights, koji su navedeni na njegovoj zvaničnoj stranici.
Ne objavljuj privatne AWS identifikatore i pune lokalne putanje iz logova bez
pregleda. Alat ne objavljuje ništa samostalno.
