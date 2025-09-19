# FirstICU Preprocessing Pipeline

This folder contains the preprocessing pipeline for extracting patient data during the **first ICU admission** from the MIMIC-IV database.  
The pipeline generates structured datasets for **MedM2T Task 2 (In-hospital Mortality Prediction)** and **Task 3 (Length of Stay Prediction)**.

---

## Required Input Files

The following raw CSV files from the [MIMIC-IV database](https://physionet.org/content/mimiciv/2.2/) are required.  
For testing, you can also use the [MIMIC-IV demo dataset](https://physionet.org/content/mimic-iv-demo/2.2/).

- **d_labitems.csv** (hospital module)  
  Item definitions for `labevents`

- **d_items.csv** (ICU module)  
  Item definitions for `chartevents`

- **labevents.csv** (hospital module)  
  Raw laboratory events

- **chartevents.csv** (ICU module)  
  Raw ICU charted events (vital signs, labs, respiratory, etc.)

Place these files inside your designated **`mimiciv_csv`** folder.  

---

## Output Files
The pipeline produces **five CSV files**:

1. **patients.csv**  
   - Patient demographics  
   - Hospital admission and discharge times  
   - ICU admission and discharge times  
   - In-hospital mortality indicator and time of death  

2. **discharge_diagnoses.csv**  
   - ICD codes for in-hospital discharge diagnoses  

3. **laboratory_measurements.csv**  
   - Laboratory test results from *labevents* and *chartevents (labs)* during ICU stay  
   - Items with more than **80% missing values** are excluded  

4. **vitalsigns.csv**  
   - Vital sign measurements from *chartevents (vitalsigns)* during ICU stay  
   - Items with more than **80% missing values** are excluded  
   - Additional respiratory indicators included: **SpO₂, FiO₂, Respiratory Rate**  

5. **items_info.csv**  
   - Metadata describing the items used in `laboratory_measurements.csv` and `vitalsigns.csv`  

---

## Data Processing Scripts

### `mimic_first_icu.py`
- Extracts **first ICU admission information** for each patient.  
- Outputs:
  - `patients.csv`  
  - `discharge_diagnoses.csv`  

---

### `mimic_event_preprocessing.py`
- Preprocesses laboratory and vital signs events.  
- Outputs:
  - `laboratory_measurements.csv`  
  - `vitalsigns.csv`  
  - `items_info.csv`  

#### Workflow

1. **init**  
   - Configure paths:  
     - `data_path`: folder for intermediate/preprocessed files  
     - `mimiciv_csv_path`: folder containing raw MIMIC-IV CSV files  

2. **load**  
   - Load `d_labitems.csv` and `d_items.csv` for item definitions.  
   - Check for cached `.plk` files in `data_path`.  
   - If not available, read from `labevents.csv` and/or `chartevents.csv`.  

3. **processing**  
   - Parse laboratory events using `labevents_value_mapping`, excluding entries listed in `labevents_excluded`.  
   - Merge overlapping items from `labevents` (labs) and `chartevents` (chartlabs).  
   - Merge `chartevents (vitalsigns)` and `chartevents (respiratory)` items.  
   - Handle overlaps: if `charttime` and `value` are identical, keep only one record.  
   - Remove null values and duplicate entries (same `charttime` + `value`).  
   - Special handling for **Ectopy events** (`itemid` = 224650, 226479, 224651, 226480):  
     - Multiple ectopy types may occur at the same time.  
     - Re-encode them as `224650_[n]` to distinguish event types.  

4. **export**  
   - Save processed results as:  
     - `laboratory_measurements.csv`  
     - `vitalsigns.csv`  
   - Export metadata about all retained items to `items_info.csv`.  

---

## Notes
- This pipeline does **not** include or release any raw MIMIC-IV data.  
- Users must have authorized access to [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) to reproduce these datasets.  
- Accessing MIMIC-IV requires completing the required training and obtaining approval via the [PhysioNet Credentialing Process](https://physionet.org/about/database/).  
