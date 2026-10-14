# Hiányzó adatok hatása és kezelése

Gépi tanulás tárgy beadandó. A projekt azt vizsgálja, hogyan rontja a hiányzó adat a gépi tanulási modellek teljesítményét, melyik pótlási (imputációs) módszer működik jól melyik modellel, és van-e kölcsönhatás az imputációs módszer és a modell között.

## Kutatási kérdések

1. Mennyit romlik a modellek teljesítménye, ha a hiányarány 5%-ról 40%-ra nő?
2. Melyik imputációs módszer a legjobb az egyes hiányarányoknál és hiánytípusoknál?
3. Van-e kölcsönhatás az imputáció és a modell között, vagyis ugyanaz a módszer más modellnél más eredményt ad-e?
4. Összefügg-e az imputált értékek pontossága a modell végső teljesítményével?

## Az adathalmaz

A kísérlet a "Student Alcohol Consumption" adathalmaz portugál nyelv kurzushoz tartozó részére épül (`data/student-por.csv`). Egy portugál középiskolai felmérés adatait tartalmazza (P. Cortez és A. Silva, 2008), 649 diák adatával és 33 oszloppal. Az oszlopok vegyes típusúak: numerikus (pl. `age`, `absences`), ordinális (pl. `studytime`, `Dalc`) és kategorikus (pl. `sex`, `Mjob`, `internet`).

- **Célváltozó:** `G3`, a diák végső jegye (0–20), tehát a feladat regresszió.
- **Hiány az eredeti adatban:** nincs. Ezért a kísérlet maga viszi be a hiányt, így az eredeti érték mindig ismert, és az imputáció pontossága is mérhető.
- **`G1` és `G2`:** a korábbi időszakok jegyei, a `G3`-mal szorosan összefüggnek. A kísérlet minden beállítást két változatban futtat: a `G1`/`G2` jellemzők között szerepelnek, illetve nem.

## A kísérlet felépítése

A kísérlet az alábbi tényezők minden kombinációját lefuttatja:

| Tényező | Értékek |
| --- | --- |
| Jellemzők | `G1`/`G2`-vel és nélkülük |
| Hiánytípus | MCAR (teljesen véletlen), MAR (az életkortól függő hiány) |
| Hiányarány | 5%, 10%, 20%, 30%, 40% |
| Imputációs módszer | átlag/módusz, KNN, iteratív |
| Modell | Ridge-regresszió, random forest, gradient boosting |
| Ismétlés | 10 különböző véletlen seed |

Minden kombináció mellett lefut egy hiány nélküli alapeset is, ez adja a felső korlátot.

Fontos tervezési szabályok:

- A hiány csak a jellemzők közé kerül, a célváltozó (`G3`) mindig teljes marad, mert ahhoz mérjük a modell hibáját.
- Az `age` oszlop soha nem kap hiányt, mert a MAR mechanizmusnál ez vezérli a hiány valószínűségét. A MCAR esetén is kimarad, hogy a két hiánytípus ugyanarra az oszlophalmazra legyen összevethető.
- A kért hiányarányt a generátor pontosan teljesíti (pl. 20% kérésnél pontosan a cellák 20%-a esik ki).
- Az imputer csak a tanítóhalmazból tanul, a teszthalmaz értékei nem szivárognak be (ezt a `scikit-learn` `Pipeline` biztosítja).
- Egy adott seednél minden módszer és modell ugyanazt a hiányt és ugyanazt a 80/20 arányú tanító/teszt felosztást kapja, így a különbségek a módszerekből adódnak.
- Minden ismétlés külön véletlen seedet használ, így a futások reprodukálhatók.

## Mérőszámok

| Mérőszám | Jelentés |
| --- | --- |
| `rmse` | a modell átlagos tévedése a `G3`-ra, jegypontban (kisebb a jobb) |
| `r2` | a `G3` ingadozásának mekkora részét magyarázza a modell (1 a tökéletes) |
| `imp_nrmse` | az imputált számértékek hibája az oszlop szórásának többszöröseként (1,0 = annyit téved, mintha az átlagot írnánk be; kisebb a jobb) |
| `imp_acc` | a kategorikus oszlopoknál az eltalált értékek aránya (nagyobb a jobb) |

## A repó felépítése

```
hianyzo-adatok/
├── config/
│   └── config.json            # a kísérlet beállításai
├── data/
│   └── student-por.csv        # az eredeti adathalmaz
├── src/
│   ├── data_loading.py        # az adathalmaz betöltése
│   ├── missingness.py         # hiánygenerátor
│   ├── imputers.py            # kódolás, imputációs módszerek
│   ├── models.py              # a modellek és a teljes feldolgozási lánc
│   ├── evaluate.py            # egy kísérlet lefuttatása és kiértékelése
│   └── run_experiments.py     # a teljes kísérletsorozat
├── notebooks/
│   ├── 01_eda.ipynb           # feltáró adatelemzés
│   └── 02_analysis.ipynb      # az eredmények elemzése, ábrák
├── results/                   # eredménytáblázat és ábrák
├── requirements.txt
└── README.md
```

### Mit csinálnak az egyes fájlok

- **`config/config.json`**: a kísérlet összes beállítása egy helyen: az ismétlések száma, a `G1`/`G2` változatai, a hiánytípusok, a hiányarányok, az imputációs módszerek, a modellek és az eredményfájl helye. A kísérlet módosításához elég ezt a fájlt átírni, a kódhoz nem kell nyúlni.
- **`src/data_loading.py`**: a `load_data()` függvény beolvassa a `data/student-por.csv` fájlt egy `pandas` táblába.
- **`src/missingness.py`**: az `inject_missing(df, rate, mechanism, seed=...)` függvény kontrolláltan hiányt visz be a jellemzők közé. MCAR esetén minden cella egyforma eséllyel esik ki, MAR esetén az idősebb diákok celláinak nagyobb az esélye kiesni. Visszaadja a hiányos táblát és egy maszkot is, amely megmutatja, mely cellákat törölte. Közvetlenül futtatva önellenőrzést végez: kiírja a kért és a tényleges hiányarányt, és ellenőrzi, hogy a célváltozó érintetlen, valamint hogy ugyanaz a seed ugyanazt a hiányt adja.
- **`src/imputers.py`**: a `build_preprocessor()` összerakja az előfeldolgozó láncot. Lépései: a szöveges oszlopok számkódolása, a hiány pótlása, a kategóriakódok egészre kerekítése (a KNN és az iteratív módszer tört számot is adhat), végül a kategóriák 0/1 oszlopokra bontása (one-hot). Az imputációs módszerek: átlag/módusz (`SimpleImputer`: számoknál átlag, kategóriáknál a leggyakoribb érték), KNN (`KNNImputer`, 5 szomszéddal) és iteratív (`IterativeImputer`, MICE-jellegű).
- **`src/models.py`**: a `build_pipeline()` az előfeldolgozást és a modellt egyetlen `Pipeline`-ba fűzi. A modellek: Ridge-regresszió (skálázással), random forest (200 fa) és gradient boosting.
- **`src/evaluate.py`**: a `run_once()` egyetlen kísérletet futtat le (hiány bevitele, 80/20 felosztás, tanítás, kiértékelés a tiszta `G3`-ra), és visszaadja a fenti mérőszámokat. Az imputáció pontosságát a törölt cellák eredeti értékéhez méri.
- **`src/run_experiments.py`**: beolvassa a `config/config.json` fájlt (és ellenőrzi, hogy nincs-e benne ismeretlen érték), összeállítja az összes kombinációt, lefuttatja őket, és az eredményt a `results/results.csv` fájlba menti (a haladást kiírja, és 50 futásonként menti a részeredményt). Parancssori kapcsolók: `--config <fájl>` másik konfigurációs fájlhoz, `--seeds <szám>` az ismétlések számának átírásához.
- **`notebooks/01_eda.ipynb`**: az adat megismerése (eloszlások, összefüggések, kategóriák).
- **`notebooks/02_analysis.ipynb`**: a `results/results.csv` feldolgozása, táblázatok és ábrák a kutatási kérdésekhez. Az ábrákat a `results/` mappába menti.

## Telepítés

Szükséges: Python 3 (a fejlesztés Python 3.14-gyel történt).

```powershell
git clone <a repó címe>
cd hianyzo-adatok
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linuxon és macOS-en az aktiválás: `source .venv/bin/activate`.

## Futtatás

Minden parancsot a projekt gyökeréből, aktivált virtuális környezetben kell kiadni.

1. **A hiánygenerátor ellenőrzése** (másodpercek):

   ```powershell
   python -m src.missingness
   ```

2. **Gyors próbafutás** egyetlen ismétléssel (néhány perc):

   ```powershell
   python -m src.run_experiments --seeds 1
   ```

3. **A teljes kísérlet** 10 ismétléssel (az alapértelmezett, a számítógéptől függően több tíz percig tart):

   ```powershell
   python -m src.run_experiments
   ```

   Az alapértelmezett beállításokat a `config/config.json` tartalmazza. Az eredmény a `results/results.csv` fájlba kerül (minden futás egy sor, a konfiguráció és a mérőszámok oszlopaival). A fájl minden futtatásnál felülíródik.

4. **Az elemzés:** `notebooks/02_analysis.ipynb`

## Eredmények

Az eredményeket, az ábrákat és a következtetéseket a külön dokumentáció tartalmazza. Az elemző notebook az ábrákat a `results/` mappába is elmenti.

## Felhasznált eszközök

Python 3, `pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `seaborn`, Jupyter.

## Forrás

P. Cortez és A. Silva: *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito és J. Teixeira (szerk.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), 5–12. o., Porto, Portugália, 2008. Az adathalmaz elérhető a Kaggle-n és az UCI Machine Learning Repository-ban.