# 🏎️ F1 Tyre Degradation Prediction

Projekat za predviđanje degradacije guma u Formuli 1 korišćenjem mašinskog učenja.  
Degradacija je definisana kao razlika između vremena posmatranog kruga i vremena drugog 
kruga u okviru istog stinta, koji se koristi kao referentni krug.
Podaci su prikupljeni putem **FastF1** biblioteke za sezone 2023, 2024 i 2025.

---

## 📋 Sadržaj

- [Zahtevi](#zahtevi)
- [Instalacija](#instalacija)
- [Struktura projekta](#struktura-projekta)
- [Redosled pokretanja](#redosled-pokretanja)
- [Pokretanje aplikacije](#pokretanje-aplikacije)
- [Napomene](#napomene)

---

## Zahtevi

- Python **3.9+**
- pip

---

## Instalacija

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/Ninus-29/f1-projekat.git
cd f1-projekat
```

### 2. Kreiranje virtualnog okruženja (preporučeno)

```bash
python -m venv venv
```

Aktivacija:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalacija zavisnosti

```bash
pip install fastf1 pandas numpy scikit-learn xgboost matplotlib streamlit joblib
```

---

## Struktura projekta

```
f1-projekat/
│
├── create_dataset.py
├── preprocess_dataset.py
├── check_seasons.py
│
├── eda.py
├── train_models.py
├── hyperparameter_tuning.py
├── feature_importance.py
├── top_features_model.py
├── prediction_analysis.py
├── model_comparison.py
│
├── encode_dataset.py
├── export_model.py
│
├── app.py
│
├── F1_Tyre_Degradation_Dokumentacija.docx
│
├── cache/
├── .gitignore
└── README.md
```

---

## Redosled pokretanja

Skripte je potrebno pokrenuti sledećim redosledom:

### Korak 1 — Prikupljanje podataka

> ⚠️ Ovaj korak traje dugo (nekoliko sati) jer preuzima podatke za sve trke.  
> Ako već imaš `f1_laps_2023_2025_raw.csv`, preskoči na korak 2.

```bash
python create_dataset.py
```

Rezultat: `f1_laps_2023_2025_raw.csv`

---

### Korak 2 — Preprocesiranje

```bash
python preprocess_dataset.py
```

Rezultat: `f1_tyre_degradation_dataset.csv`

---

### Korak 3 — Provera dataseta *(opcionalno)*

```bash
python check_seasons.py
```

Ispisuje broj redova po sezoni i broj trka.

---

### Korak 4 — Eksplorativna analiza *(opcionalno)*

```bash
python eda.py
```

Rezultat: `degradation_distribution.png` i statistike u konzoli.

---

### Korak 5 — Treniranje modela

```bash
python train_models.py
```

Rezultat: `model_results.csv`

---

### Korak 6 — Poređenje modela *(opcionalno)*

```bash
python model_comparison.py
```

Rezultat: `model_comparison.png`

---

### Korak 7 — Podešavanje hiperparametara *(opcionalno)*

> ⚠️ Može trajati duže u zavisnosti od mašine.

```bash
python hyperparameter_tuning.py
```

Rezultat: `hyperparameter_tuning_results.csv`

---

### Korak 8 — Analiza važnosti atributa *(opcionalno)*

```bash
python feature_importance.py
```

Rezultat: `top20_features.csv`, `feature_importance.png`

---

### Korak 9 — Poređenje: svi vs. top 5 atributa *(opcionalno)*

```bash
python top_features_model.py
```

Rezultat: `top_features_comparison.csv`

---

### Korak 10 — Vizualizacija predikcija *(opcionalno)*

```bash
python prediction_analysis.py
```

Rezultat: `actual_vs_predicted.png`

---

### Korak 11 — Enkodovanje dataseta

```bash
python encode_dataset.py
```

Rezultat: `f1_tyre_degradation_encoded.csv`

---

### Korak 12 — Export finalnog modela

```bash
python export_model.py
```

Rezultat: `f1_tyre_degradation_model.pkl`

---

## Pokretanje aplikacije

Nakon što je model eksportovan (`f1_tyre_degradation_model.pkl` mora postojati):

```bash
streamlit run app.py
```

Aplikacija se otvara u browseru na adresi `http://localhost:8501`.

### Režimi rada

**Validate Existing Data** — Biraš trku, vozača, stint i broj kruga iz dataseta. Aplikacija poredi predikciju modela sa stvarnom vrednošću degradacije.

**Predict New Scenario** — Unosiš novu kombinaciju parametara. Za vrednosti koje ne unosiš ručno (vremenski uslovi) koriste se prosečne vrednosti iz dataseta za odabranu trku.

---

## Napomene

- Fajlovi navedeni u `.gitignore` se ne commituju (sirovi CSV-ovi, keš, `.pkl` model)
- FastF1 keš se čuva u folderu `cache/` i ubrzava ponovljeno preuzimanje podataka
- Sezona 2025 je u datasetu zastupljena sa samo 3 trke (podaci dostupni u trenutku prikupljanja)
- Za reprodukciju rezultata koristiti `random_state=42` (već postavljeno u svim skriptama)
