# Fő elemzés (`elemzes` branch)

A hiányzó adatos kísérlet mellett a projekt tartalmaz egy általános, teljes körű gépi tanulási elemzést is, a hiány nélküli (eredeti) adathalmazon. Feladat: a `G3` végső jegy becslése (regresszió). Az elemzés két változatban fut: a `G1`/`G2` jellemzőkkel és nélkülük.

## Mit tartalmaz

| Lépés | Hol |
| --- | --- |
| EDA, hiányzó értékek, outlierek, skálázás, kódolás | `notebooks/03_elemzes_eda.ipynb`, `src/elemzes/features.py`, `src/elemzes/pipelines.py` |
| Feature engineering, PCA és jellemzőkiválasztás | `src/elemzes/features.py`, `src/elemzes/pipelines.py`, az "A kísérlet" a `run_analysis.py`-ban |
| Adatszivárgás-mentes, beágyazott K-Fold CV, baseline és haladó modell | `src/elemzes/validation.py`, `src/elemzes/run_analysis.py` |
| Mutatók, konfidenciaintervallumok, statisztikai próba | `src/elemzes/validation.py`, `notebooks/04_elemzes_eredmenyek.ipynb` |
| Jellemzőfontosság (csoportos permutációs fontosság, SHAP) és korlátai | `src/elemzes/importance.py`, `notebooks/04_elemzes_eredmenyek.ipynb` |

## A fájlok

- `config/elemzes.json`: a beállítások (seed, a CV szerkezete, a változatok, a kiválasztott jellemzők száma, az eredmények helye).
- `src/elemzes/features.py`: a `FeatureBuilder` (új jellemzők: heti alkoholfogyasztás, a szülők végzettségének összege, társasági index, támogatások száma, jegyváltozás) és az `OutlierClipper` (a kilógó értékek vágása a tanítóhalmazból számolt Tukey-kerítésre). Mindkettő scikit-learn transzformer, így a lánc részeként csak a tanítóhalmazból tanul.
- `src/elemzes/pipelines.py`: a teljes lánc (`build_pipeline`): jellemzőkészítés, outliervágás, imputálás (biztonsági háló, az eredeti adatban nincs hiány), standardizálás, one-hot kódolás, dimenziócsökkentés (nincs / PCA / 20 legjobb jellemző) és modell. A modellek: átlagjósló (alsó referencia), Ridge-regresszió (baseline) és gradient boosting (haladó). Itt vannak a hiperparaméter-rácsok is.
- `src/elemzes/validation.py`: a külső ismételt K-Fold, a belső hangoló CV, a korrigált (Nadeau-Bengio) konfidenciaintervallum, a baseline és a haladó modell páros összehasonlítása (korrigált t-próba, kiegészítésként Wilcoxon).
- `src/elemzes/importance.py`: csoportos permutációs fontosság a tanításhoz nem használt adaton, a jellemzőnevek kinyerése és a leggyakoribb hiperparaméterek kiválasztása.
- `src/elemzes/plots.py`: az ábrák.
- `src/elemzes/run_analysis.py`: az egész elemzés lefuttatása, az eredmények és ábrák mentése a `results/elemzes/` mappába.
- `notebooks/03_elemzes_eda.ipynb`: EDA, outlierek, feature engineering, PCA.
- `notebooks/04_elemzes_eredmenyek.ipynb`: az eredmények táblázatai és ábrái.

## Módszertani szabályok

- Az imputálás, outliervágás, skálázás, kódolás, jellemzőkiválasztás és PCA mind a `Pipeline`-ban van, ezért a CV minden körében csak a tanítóhalmazból tanul: nincs adatszivárgás.
- A hiperparaméter-hangolás beágyazott: a külső CV minden tanítóhalmazán külön fut a belső 3-szoros CV, a külső teszthalmaz soha nem szól bele.
- Minden modell ugyanazokat a külső felosztásokat kapja (5-szörös K-Fold, 5 ismétlés, rögzített seed), így a párosított összehasonlítás jogos.
- Az ismételt CV körei nem függetlenek, ezért a konfidenciaintervallumok és az elsődleges próba a Nadeau-Bengio-féle korrigált szórást használják.
- A PCA-t a főkomponensek értelmezhetetlensége miatt kizárjuk a végső láncból, a jellemzők számát a "nincs csökkentés" és a "20 legjobb jellemző" közül az A kísérlet átlagos RMSE-je dönti el.
- A permutációs fontosságot a külső CV tesztfoldjain számoljuk (a modell nem látta ezeket), a SHAP a végleges modellen, a teljes adaton készül.

## Futtatás

A projekt gyökeréből, aktivált virtuális környezetben (előtte egyszer: `pip install -r requirements.txt`):

```powershell
python -m src.elemzes.run_analysis --repeats 1    # gyors próbafutás (kb. 2 perc)
python -m src.elemzes.run_analysis                # a teljes elemzés (kb. 5-6 perc)
```

Az eredmény a `results/elemzes/` mappába kerül (CSV táblázatok és PNG ábrák). Utána futtathatók a notebookok.