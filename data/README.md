# Data

## Source

Ogbuokiri, B. (2023): *Sensor-Based Aquaponics Fish Pond Datasets*. Kaggle.
<https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets>

Data collection is described in: Ogbuokiri, B. O., Ossai, N. I., Nweke, O., and Udanor, C. (2022): A Pilot
Implementation of a Remote IoT Sensors for Aquaponics System Datasets Acquisition. *Journal of Computer Science and
Its Application*, Nigeria Computer Society, 28(2).

Twelve catfish aquaponics ponds, each with an ESP32 unit and six sensors (DS18B20 temperature, DFRobot turbidity,
DFRobot dissolved oxygen, DFRobot pH V2.2, MQ-137 ammonia, MQ-135 nitrate), plus fish length and weight recorded on
sampling days, from June 2021. Pond 5 has no CSV, so 11 ponds are used (`IoTPond1.csv` … `IoTPond12.csv`).

> **Licence:** check the licence shown on the Kaggle page before redistributing any file, and keep the citation
> above. Only one small pond is included here as a sample.

## Layout

| Folder | Content | In git |
| --- | --- | --- |
| `data/raw/` | The 11 original pond CSVs, unchanged | No (≈1.1 M rows) |
| `data/processed/` | Written by `notebooks/03_preprocessing.ipynb` | No |
| `data/sample/IoTPond10.csv` | One small pond (620 rows) for a quick look | Yes |

To reproduce the analysis, download the pond CSVs into `data/raw/` and run the notebooks in order.

## Known quirks handled in `src/pond_pipeline.py`

- Units are mg/L; some exports label oxygen, ammonia and nitrate as `g/ml`. Columns are renamed on load.
- Timestamps mix `2021-06-25 15:48:59 CET` and day-first `7/11/2021` (7 November). Both formats are parsed explicitly.
- Some exports store nitrate ×100 (193 instead of 1.93 mg/L). Detected by the median and rescaled.
- Sensor error codes such as −127 °C and ammonia of 4.27e11 or `inf` are removed by the valid-range check.
- `Fish_Length(cm)` in pond 4 contains `Jan-00` (an Excel artefact), treated as missing.
- Ponds 7, 8 and 9 have no population column; values 50, 75 and 50 are set from the thesis records.
