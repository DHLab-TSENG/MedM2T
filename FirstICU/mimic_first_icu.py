import pandas as pd
import pymysql

db_host = "..."
db_user = "..."
db_passwd = "..."
db_name = "mimiciv"
out_dir = "./mimic_data"

mimiciv_db = pymysql.connect(host=db_host, user=db_user, passwd=db_passwd, db=db_name)
mimiciv = mimiciv_db.cursor()

first_icustay_sql = "SELECT icu.* FROM icustays icu  \
                    JOIN ( SELECT subject_id, MIN(intime) AS first_intime FROM icustays GROUP BY subject_id \
                    ) AS first_icu ON icu.subject_id = first_icu.subject_id AND icu.intime = first_icu.first_intime;"
                        
mimiciv.execute(first_icustay_sql)
first_icu = mimiciv.fetchall()
first_icu_df = pd.DataFrame(first_icu, columns = ["subject_id","hadm_id","stay_id","first_careunit","last_careunit","icu_intime","icu_outtime","icustay_days"])
first_icu_df.to_csv("%s/first_icu.csv"%(out_dir),index = False)

first_icu_admission_sql = "SELECT * FROM admissions WHERE hadm_id IN (SELECT hadm_id FROM first_icustay);"
mimiciv.execute(first_icu_admission_sql)
first_icu_admission = mimiciv.fetchall()
first_icu_admission_df = pd.DataFrame(first_icu_admission, columns = ["subject_id","hadm_id","admittime","dischtime","deathtime","admission_type","admit_provider_id","admission_location","discharge_location","insurance","language","marital_status","race","edregtime","edouttime","hospital_expire_flag"])

first_icu_patients_sql = "SELECT * FROM patients WHERE subject_id IN (SELECT subject_id FROM first_icustay);"
mimiciv.execute(first_icu_patients_sql)
first_icu_patients = mimiciv.fetchall()
first_icu_patients_df = pd.DataFrame(first_icu_patients, columns = ["subject_id","gender","anchor_age","anchor_year","anchor_year_group","dod"])

patients_df = first_icu_df.merge(first_icu_admission_df, on=["subject_id","hadm_id"])
patients_df = patients_df.merge(first_icu_patients_df, on=["subject_id"])
patients_df = patients_df[["subject_id","gender","anchor_age","hadm_id","stay_id","admittime","dischtime","deathtime","admission_type","admission_location","hospital_expire_flag","first_careunit","last_careunit","icu_intime","icu_outtime","icustay_days"]]
patients_df.to_csv("%s/patients.csv"%(out_dir),index = False)

first_icu_diagnoses_sql = "SELECT * FROM diagnoses_icd WHERE hadm_id IN (SELECT hadm_id FROM first_icustay);"
mimiciv.execute(first_icu_diagnoses_sql)
first_icu_diagnoses = mimiciv.fetchall()
first_icu_diagnoses_df = pd.DataFrame(first_icu_diagnoses, columns = ["subject_id","hadm_id","seq_num","icd_version","icd_code"])
first_icu_diagnoses_df.to_csv("%s/discharge_diagnoses.csv"%(out_dir),index = False)

