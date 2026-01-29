# Learning-Analytics: Himawari-8 Rainfall Data Analysis & Prediction

Un pipeline complet de **data engineering și machine learning** care descarcă date satelitare Himawari-8 din Google Cloud, le transformă în serii temporale meteorologice folosind PySpark și antrenează mai multe modele de regresie pentru a **prezice rata precipitațiilor**.

Proiectul este conceput ca un studiu de **Learning Analytics & Big Data** aplicat în meteorologie, folosind date reale NOAA/JAXA și procesare distribuită.

## 🚀 Key Features

* **NOAA Cloud Integration**  
  Descărcare automată a fișierelor NetCDF de la:
  `gs://noaa-himawari8/AHI-L2-FLDK-RainfallRate`

* **Satellite Data Engineering**  
  Transformarea imaginilor satelitare 2D în **indicatori statistici de precipitații**:
  * medie (mean)
  * maxim (max)
  * deviație standard (std)
  * fracția de pixeli ploioși

* **Spark Feature Engineering**  
  Generare de caracteristici temporale folosind PySpark:
  * codare ciclică orară (sin / cos)
  * diferențe temporale
  * valori întârziate (lag 1, 2, 3)
  * caracteristici autoregresive

* **Multi-Model Training**  
  Antrenare și comparare a patru modele:
  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * XGBoost (Spark)

* **Automated Evaluation**  
  Generare automată de:
  * RMSE și R²
  * grafice time-series
  * scatter plots (predicted vs real)
  * histograme ale erorilor
  * importanța caracteristicilor



## 📋 Prerequisites

* **Python 3.8+**
* **Java 8 sau 11** (necesar pentru Spark)
* **Google Cloud SDK (`gsutil`)**
* Acces la internet (pentru NOAA Cloud)

---

## 🛠️ Instalare

1. **Clonează repository-ul**
```bash
git clone https://github.com/RaDuOnT/Learning-Analytics-Himawari.git
cd Learning-Analytics-Himawari
```

2. **Instalează dependențele**

```bash
pip install pyspark numpy pandas xarray netCDF4 matplotlib xgboost
```

3. **Configurează Google Cloud SDK**

```bash
gsutil ls
```

Dacă funcționează, accesul la NOAA bucket este valid.

---

## 📦 Pipeline Usage

Pipeline-ul poate fi rulat fie etapizat, fie cap-coadă.

---

### Step 1 – Descărcare date satelitare

Descarcă datele NetCDF (sub-eșantionate orar):

```bash
python download_himawari.py
```

---

### Step 2 – Procesare și extragere caracteristici

Transformă imaginile satelitare în serii temporale:

```bash
python process_himawari.py
```

Rezultatul este salvat în:

```
data_parquet/himawari_rr_features.parquet
```

---

### Step 3 – Antrenare modele ML

Rulează pipeline-ul Spark pentru regresie:

```bash
python train_spark_models.py
```

Modelele sunt salvate în:

```
models/
```

---

### Step 4 – Generare grafice și evaluare

```bash
python make_plots.py
```

Rezultatele sunt generate în:

```
plots/
```

---

## 📂 Project Structure

```
.
├── download_himawari.py      # Download NOAA Himawari-8 data
├── process_himawari.py       # NetCDF → Parquet + Feature Engineering
├── train_spark_models.py    # PySpark ML pipeline
├── make_plots.py            # Evaluation & visualizations
├── data_raw_2020/           # Raw satellite files
├── data_parquet/            # Feature dataset
├── models/                  # Trained models
└── plots/                   # Graphs & metrics
```

---

## 🧠 Machine Learning Design

Modelele folosesc:

**Variabile de intrare**

* max_rr, std_rr, frac_rainy
* delta_minutes
* hour_sin, hour_cos
* mean_rr_lag1, mean_rr_lag2, mean_rr_lag3

**Variabila țintă**

```
mean_rr (rata medie de precipitații)
```

**Split**

* Train: anul 2020
* Test: 20–27 iunie 2021

---

## 📊 Output

Pipeline-ul produce:

* Predicții vs valori reale
* RMSE & R² per model
* Importanță caracteristici
* Analiză erori

Toate sunt salvate automat în `plots/`.

---

## ⚠️ Troubleshooting

### Spark out of memory

Editează în `train_spark_models.py`:

```python
.config("spark.driver.memory", "4g")
```

### XGBoost GPU

```python
USE_GPU_FOR_XGB = True
```

Necesită CUDA + XGBoost Spark cu suport GPU.

---

## 📌 Scop academic

Acest proiect demonstrează cum **Big Data, Cloud Computing și Machine Learning** pot fi integrate într-un sistem de analiză meteorologică reală folosind date satelitare.

Este ideal pentru:

* Learning Analytics
* Data Engineering
* Big Data cu PySpark
* Time Series Forecasting
* Climate & Weather AI
