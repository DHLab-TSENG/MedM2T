# A. DATASET
## I. TASK 1: CVD DATASET
### Population selection
**[Fig. A1](#fig-a1)** illustrates the population selection process for the CVD dataset used in Task 1. 
Patients were included if they had at least one hospitalization occurring within 90 days after an ECG measurement. 
Patients were excluded if they were under 18 years of age or over 89 years of age, or if their hospital stays were shorter than 24 hours. 
Records with admissions containing CVD diagnoses but without CVD as the primary diagnosis or operation were excluded, as the cause of admission could not be confirmed to be CVD-related.  
After applying these criteria, the dataset comprised 44,790 CVD-related ECG samples from 13,289 patients, further categorized into coronary heart disease (CHD, N=18,445), stroke (N=4,927), and heart failure (HF, N=21,418). In addition, 125,987 non-CVD ECG samples from 52,388 patients were identified. 
The ICD code definitions for each type of CVD are summarized in **Table A1** [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12].  
  
<img alt="image" src="https://github.com/user-attachments/assets/0a2db664-7d34-4118-ba94-ba9016795d20" style="width:800px;"/>  

#### **Fig. A1.** 
> Population selection process for Task 1 (CVD dataset). Focusing on patients with at least one hospitalization occurring within 90 days after an ECG measurement. The final dataset included 44,790 CVD-related and 125,987 non-CVD ECG samples.  

### Data description
1. **EHR Static Data:** Patient demographics included gender and age, extracted from the patient table. Latest outpatient measurements (systolic/diastolic blood pressure, weight, and height) were taken from the omr table. Medical history was defined based on ten CVD-related conditions reported in the literature [13], [14], [15], [16], with detailed definitions listed in **Table A1**. Medication history was derived from the prescriptions table, in which drugs were mapped to Anatomical Therapeutic Chemical (ATC) codes using the RxNorm API [17]. We selected ATC first-level class C (cardiovascular system) and grouped drugs by their third-level categories (e.g., C01A).  
Detailed variable summaries are provided in [CVD_Static.csv](DataDescriptions/CVD_Static.csv), and the codebook is available at [Codebook.md](DataDescriptions/Codebook.md).

2. **Laboratory Results:** Eight laboratory tests relevant to CVD were selected according to prior studies [13], [14], [15], [16], including estimated glomerular filtration rate (eGFR), troponin T, creatine kinase, creatine kinase-MB, serum creatinine, high-density lipoprotein (HDL) cholesterol, low-density lipoprotein (LDL) cholesterol, and total cholesterol. These values were obtained from the labevents table.  
Detailed variable summaries are provided in [CVD_Labs.csv](DataDescriptions/CVD_Labs.csv), with the corresponding codebook in [Codebook.md](DataDescriptions/Codebook.md).

3. **ECG Signals, Text, and Features:** Raw 12-lead ECG recordings (500 Hz, 10 s) underwent the following preprocessing steps: (1) interpolation of missing values, (2) down-sampling to 125 Hz, (3) removal of noise and baseline wander, (4) application of a third-order Butterworth band-pass filter (0.5–40 Hz), and (5) segmentation into 5-second windows.  
Besides raw signals, nine time-domain features (e.g., heart rate, PR interval) were got from machine-generated ECG reports. Detailed variable summaries are provided in [CVD_ECG.csv](DataDescriptions/CVD_ECG.csv).  
Machine-generated ECG reports were further preprocessed and mapped to 143 SNOMED CT clinical terms [18], providing structured and interpretable diagnostic judgments. Examples of the mapping are shown in **Fig. A2**, and the distribution of mapped samples is summarized in [ECG_Notes.csv](DataDescriptions/ECG_Notes.csv).

<img alt="image" src="https://github.com/user-attachments/assets/9b19845e-03b2-49e7-9adf-0802292f677b" style="width:800px;"/>

#### **Fig. A2.** 
> Examples of machine-generated ECG reports mapped to SNOMED CT clinical terms. Highlighted terms indicate structured mappings such as atrial flutter, atrioventricular (AV) block, and premature ventricular complexes (PVCs), which facilitate interpretable representation of diagnostic findings.

#### **TABLE A1.** 
> DEFINITIONS OF CVD CATEGORY AND MEDICAL HISTORY LABEL MAPPINGS
<img alt="image" src="https://github.com/user-attachments/assets/9d02dbdd-c320-416e-8f16-7b4f172fc14c" style="width:700px;"/>

## II. TASK 2 & 3: MORTALITY AND LENGTH-OF-STAY (LOS) DATASET
### Population selection
**Fig. A3** shows the population selection process for the Mortality dataset. We included only the patient’s first ICU admission and excluded records with ICU stays shorter than 24 hours. After applying these criteria, the dataset contained 40,167 records, of which 4,035 (10.04%) corresponded to in-hospital deaths.  
The LOS dataset employed the same population as the Mortality dataset, with the length of ICU stay calculated for each patient. The average LOS was 4.05 days (SD = 5.19), and the distribution is shown in **Fig. A4**.

<img alt="image" src="https://github.com/user-attachments/assets/1a7bd0f7-1964-476a-8398-65ed4e744f03" style="width:800px;"/>

#### **Fig. A3.** 
> Population selection process for Task 2 (in-hospital mortality). The final cohort included 40,167 first ICU admission records, with 4,035 mortality cases (10.04%).

<img alt="image" src="https://github.com/user-attachments/assets/1ddb2000-8c97-40ec-98d0-7c0a8a52cd8f" style="width:500px;"/>

#### **Fig. A4.** 
> Distribution of ICU length of stay (LOS) for Task 3. The mean LOS was 4.05 days with a standard deviation of 5.19 days.

### Data description
1. **EHR Static Data:** Demographic information (gender, age) was extracted from the patient table, while admission details (admission type, admission location) were obtained from the admissions table.  
Detailed variable summarize are available in [Mortality_Static.csv](DataDescriptions/Mortality_Static.csv), and the codebook is provided in [Codebook.md](DataDescriptions/Codebook.md).

2. **Vital Signs:** Hourly vital sign measurements were derived from the chartevents table. Only variables with less than 80% missingness across all ICU patients were selected. Multiple itemid entries corresponding to the same variable (e.g., 220179, 220050, 224167, 227243 for systolic blood pressure) were merged into a single representative identifier (e.g., 220179).  
Detailed variable summaries are provided in [Mortality_Vitals.csv](DataDescriptions/Mortality_Vitals.csv).

3. **Laboratory Tests:** Laboratory test results were extracted from the chartevent and labevents tables. Only variables with less than 80% missingness across all ICU patients were selected. Multiple itemid entries corresponding to the same variable were merged according to test item and specimen type (e.g., 50862 and 227456 for blood albumin were merged into 50862). Units were standardized before merging, and duplicate entries recorded at the same time were removed.  
Detailed variable summaries are provided in [Mortality_Labs.csv](DataDescriptions/Mortality_Labs.csv).

4. **ECG Signals, Text, and Features:** The ECG preprocessing procedure was identical to Task 1.  
Detailed summaries of ECG features are provided in [Mortality_ECG.csv](DataDescriptions/Mortality_ECG.csv).

# B. EXPERIMENTAL SETUP
## I. HYPERPARAMETERS OF MEDM2T
### Model Hyperparameters
**[Table B1](TABLE_B1.pdf)** summarizes the model hyperparameters of MedM2T. When values differed across tasks, they are reported in the format **Task 1 / Task 2 / Task 3**.
#### Unimodal encoders:
1. **Static:** encoded by a multilayer perceptron (MLP).
2. **Labs:** encoded by the proposed sparse time-series encoder.  
3. **Vitals:** integrating categorical and numerical vitals, using the same architecture as the multimodal encoder.
  - Vitals (C): categorical vitals encoded with the sparse time-series encoder.
  - Vitals (N): numerical vitals encoded with the hierarchical time-aware fusion model. This includes multi-scale high-frequency encoders (High-Freq 1 for a window size of 12, High-Freq 2 for a window size of 24) and a low-frequency encoder (Low-Freq).
4. **ECG:** encoded by the hierarchical time-aware fusion model, which integrates ECG-level representations across multiple time points.
  - ECG (T): ECG text encoded with the embedding layer and MLP.
  - ECG (S): ECG signals encoded with ResNet and MLP.
  - ECG (F): ECG features encoded with MLP.
  - ECG (Fusion): integration of ECG (T), ECG (S), and ECG (F) into ECG-level representations, using the same architecture as the multimodal encoder.
#### Multimodal encoder:  
composed of modality-specific encoders for each input type, a shared encoder, and the Bi-Modal attention modules.
#### Decoders:  
all implemented as a one-layer MLP.

### Algorithm Hyperparameters
**[Table B2](TABLE_B2.pdf)** lists the algorithm hyperparameters used during model training. An early stopping strategy was applied, with the maximum number of epochs set to 20.

## II. HYPERPARAMETERS OF COMPARED MODELS
We compared MedM2T against several state-of-the-art multimodal frameworks, including MultiBench [19], MultiModN [20], and HAIM [21]. For MultiBench and MultiModN, most hyperparameters were adopted from their original configurations applied for the MIMIC datasets, with minor adjustments to hidden dimensions and learning rates. HAIM followed the hyperparameter tuning strategy recommended in the original work. Each modality’s input type and encoder are summarized in Table B3, where time series (stats) refers to statistical feature extraction via the HAIM framework. MultiBench and MultiModN constructed multimodal learning frameworks using encoder–fusion–decoder architectures, whereas HAIM applied preprocessing followed by XGBoost for classification and regression.   
For other compared models that did not explicitly provide hyperparameter settings, we used configurations consistent with MedM2T for similar data types and tasks, with additional tuning of learning rates.

#### **TABLE B3.** 
> MODALITY TYPE AND ENCODER OF COMPARED MULTIMODAL FRAMEWORKS
<img alt="image" src="https://github.com/user-attachments/assets/28dbcbdb-46ed-4b6c-aa8c-21b7b8c2c4b7" style="width:700px;"/>

## III. SAMPLE SIZES OF TASKS
For unimodal tasks, only the subset of samples with available data in the specific modality was used, whereas multimodal tasks utilized the full cohort. 
**Table B4** reports the number of samples for each task. For ECG-related modalities (ECG (T), ECG (S), ECG (F), and ECG (Fusion)), each ECG measurement record was treated as a separate training sample.  
Task 1 is a multiclass classification problem. **Table B5** presents the distribution of samples across CVD categories for each modality.  
ECG data were largely missing in Task 2 and Task 3, with only 46.7% sample having available ECG. In the ECG-available subset, only a few samples had over two ECG measurements in Task 2 and Task 3. 
**Table B6** summarizes the proportion of samples containing multiple ECG measurements in each task. 

#### **TABLE B4.** 
> SAMPLE SIZES OF TASKS
<img alt="image" src="https://github.com/user-attachments/assets/75d68fc1-df11-4c8d-9783-359f46e9c4e7" style="width:600px;"/>

#### **TABLE B5.** 
> SAMPLE DISTRIBUTION ACROSS CVD CATEGORIES FOR TASK 1
<img alt="image" src="https://github.com/user-attachments/assets/57417465-089d-412d-91cd-8fa288d6b79d" style="width:600px;"/>

#### **TABLE B6.** 
> PROPORTION OF SAMPLES WITH MULTIPLE ECG MEASUREMENTS
<img alt="image" src="https://github.com/user-attachments/assets/8cf7f42f-a1a0-4f26-b838-441bc86bec8e" style="width:500px;"/>

# C. EXPERIMENTAL RESULTS
## I. MEDM2T RESULT
**Table C1** presents the results of MedM2T under all unimodal and multimodal combinations across the three tasks, providing supplementary details corresponding to **Table IV** in the main manuscript.  
The distinction between multimodal combinations highlights the diverse contributions of each modality. The exclusion of laboratory tests led to the largest performance drop in Task 1 and Task 2, whereas the exclusion of vital signs caused the greatest decline in Task 3. 
Notably, for Task 2, vitals achieved the best unimodal performance, whereas the exclusion of laboratory tests caused the largest degradation, suggesting that laboratory tests provide complementary and distinctive information when combined with other modalities.

#### **TABLE C1.** 
> PERFORMANCE OF MEDM2T ACROSS UNIMODAL AND MULTIMODAL
<img alt="image" src="https://github.com/user-attachments/assets/ce0a8644-7b71-4921-ba0a-93e8f898e866" style="width:700px;"/>

**Table C2** provides the detailed results of Task 1 (CVD prediction) across four categories: non-CVD, CHD, stroke, and HF, serving as supplementary details corresponding to Fig. 5 in the main manuscript.  
For unimodal, the static extended subset (including medical history and medications) achieved substantially better performance than the core subset across all classes. 
As more modalities were integrated, the performance gap between the core and extended subsets diminished, suggesting complementary information across modalities.

#### **TABLE C2.** 
> PERFORMANCE OF MEDM2T FOR TASK 1 (CVD PREDICTION) ACROSS FOUR CATEGORIES
<img alt="image" src="https://github.com/user-attachments/assets/ff0ca0b2-040c-4110-ab0c-65fb9c552971" style="width:700px;"/>

## II. BI-MODAL ATTENTION ABLATION STUDY IN TASK 1
**Table C3** provides an ablation study of the Bi-Modal Attention module in Task 1 (CVD prediction). As reported in the main manuscript (Table V), Bi-Modal Attention enhances multimodal learning performance. 
Here, we further validate its effect across different multimodal combinations, a random subset of the multiclass task (N=50,000), and a binary classification task (CVD vs. non-CVD, N=50,000).  
In all cases, incorporating Bi-Modal Attention consistently improved both AUROC and AUPRC compared with models trained without it.

#### **TABLE C3.** 
> ABLATION STUDY OF BI-MODAL ATTENTION IN TASK 1
<img alt="image" src="https://github.com/user-attachments/assets/da7c4644-8c33-42b8-b98d-075f328bf2a8" style="width:700px;"/>

## III. PERFORMANCE OF COMPARATIVE FRAMEWORKS
**Table C4** compares the performance of MedM2T with other state-of-the-art multimodal frameworks (MultiBench, MultiModN, and HAIM) under unimodal and multimodal across the three tasks. 
This table provides supplementary details corresponding to Table V in the main manuscript. For MultiBench, MultiModN, and HAIM, we adopted the encoder configurations recommended in their original implementations. 
The input data types and encoders for each modality are summarized in **Table B3**.  
Across all three tasks, MedM2T got the best results under multimodal integration, while HAIM showed competitive performance in unimodal settings with static or laboratory data.

#### **TABLE C4.**
> COMPARISON OF MEDM2T WITH OTHER MULTIMODAL FRAMEWORKS
<img alt="image" src="https://github.com/user-attachments/assets/3bcdfd49-5a9c-498c-a635-b2f95c307d3f" style="width:700px;"/>

# REFERENCE
[1]	M. Kivimäki and A. Steptoe, “Effects of stress on the development and progression of cardiovascular disease,” Nat Rev Cardiol, vol. 15, no. 4, pp. 215–229, Apr. 2018.  
[2]	W. Bai, B. Hao, W. Meng, J. Qin, W. Xu, and L. Qin, “Association between frailty and short- and long-term mortality in patients with critical acute myocardial infarction: Results from MIMIC-IV,” Front Cardiovasc Med, vol. 9, Dec. 2022.  
[3]	C. Li et al., “Developing and verifying a multivariate model to predict the survival probability after coronary artery bypass grafting in patients with coronary atherosclerosis based on the MIMIC-III database,” Heart Lung, vol. 52, pp. 61–70, Mar. 2022.  
[4]	L. Tong et al., “Relationship between the red cell distribution width-to-platelet ratio and in-hospital mortality among critically ill patients with acute myocardial infarction: a retrospective analysis of the MIMIC-IV database,” BMJ Open, vol. 12, no. 9, Sep. 2022.  
[5]	L. Jian, Z. Zhang, Q. Zhou, X. Duan, H. Xu, and L. Ge, “Association between albumin corrected anion gap and 30-day all-cause mortality of critically ill patients with acute myocardial infarction: a retrospective analysis based on the MIMIC-IV database,” BMC Cardiovasc Disord, vol. 23, no. 1, Dec. 2023.  
[6]	T. Shu et al., “Development and assessment of scoring model for ICU stay and mortality prediction after emergency admissions in ischemic heart disease: a retrospective study of MIMIC-IV databases,” Intern Emerg Med, vol. 18, no. 2, pp. 487–497, Mar. 2023.  
[7]	J. Chen, Y. Li, P. Liu, H. Wu, and G. Su, “A nomogram to predict the in-hospital mortality of patients with congestive heart failure and chronic kidney disease,” ESC Heart Fail, vol. 9, no. 5, pp. 3167–3176, Oct. 2022.  
[8]	J. M. Schaffer et al., “Differences in Administrative Claims Data for Coronary Artery Bypass Grafting Between International Classification of Diseases, Ninth Revision and Tenth Revision Coding,” JAMA Cardiol, vol. 6, no. 9, pp. 1094–1096, Sep. 2021.  
[9]	S. Wu, X. Shi, Q. Zhou, X. Duan, X. Zhang, and H. Guo, “The Association between Systemic Immune-Inflammation Index and All-Cause Mortality in Acute Ischemic Stroke Patients: Analysis from the MIMIC-IV Database,” Emerg Med Int, vol. 2022, p. 4156489, Aug. 2022.  
[10]	H. J. Jhou, P. H. Chen, L. Y. Yang, S. H. Chang, and C. H. Lee, “Plasma Anion Gap and Risk of In-Hospital Mortality in Patients with Acute Ischemic Stroke: Analysis from the MIMIC-IV Database,” J Pers Med, vol. 11, no. 10, Oct. 2021.  
[11]	X. Ji and W. Ke, “Red blood cell distribution width and all-cause mortality in congestive heart failure patients: a retrospective cohort study based on the Mimic-III database,” Front Cardiovasc Med, vol. 10, 2023.  
[12]	M. Yuan, F. Ren, and D. Gao, “The Value of SII in Predicting the Mortality of Patients with Heart Failure,” Dis Markers, vol. 2022.  
[13]	E. A. Bohula et al., “Atherothrombotic Risk Stratification and the Efficacy and Safety of Vorapaxar in Patients with Stable Ischemic Heart Disease and Previous Myocardial Infarction,” Circulation, vol. 134, no. 4, pp. 304–313, Jul. 2016.  
[14]	K. A. A. Fox et al., “Prediction of risk of death and myocardial infarction in the six months after presentation with acute coronary syndrome: prospective multinational observational study (GRACE),” BMJ : British Medical Journal, vol. 333, no. 7578, p. 1091, Nov. 2006.  
[15]	P. W. F. Wilson et al., “An international model to predict recurrent cardiovascular disease,” American Journal of Medicine, vol. 125, no. 7, pp. 695-703.e1, Jul. 2012.  
[16]	D. De Bacquer et al., “Prediction of recurrent event in patients with coronary heart disease: the EUROASPIRE Risk Model,” Eur J Prev Cardiol, vol. 29, no. 2, pp. 328–339, Jan. 2022.  
[17]	S. J. Nelson, K. Zeng, J. Kilbourne, T. Powell, and R. Moore, “Normalized names for clinical drugs: RxNorm at 6 years,” Journal of the American Medical Informatics Association, vol. 18, no. 4, pp. 441–448, Jul. 2011.  
[18]	“SNOMED CT - NHS England Digital.” Accessed: Sep. 15, 2025. [Online]. Available: https://digital.nhs.uk/services/terminology-and-classifications/snomed-ct  
[19]	P. P. Liang et al., “MultiBench: Multiscale Benchmarks for Multimodal Representation Learning,” in Proc. Adv. Neural Inf. Process. Syst. Datasets & Benchmarks Track, 2021.  
[20]	V. Swamy et al., “MultiModN- Multimodal, Multi-Task, Interpretable Modular Networks,” in Proc. Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 28115–28138.  
[21]	L. R. Soenksen et al., “Integrated multimodal artificial intelligence framework for healthcare applications,” npj Digital Medicine 2022 5:1, vol. 5, no. 1, pp. 1–10, Sep. 2022.  

