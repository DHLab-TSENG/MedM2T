# MedM2T: A MultiModal Framework for Time-Aware Modeling with Electronic Health Record and Electrocardiogram Data
This repository contains the official implementation of MedM2T, as described in our paper:
MedM2T: A MultiModal Framework for Time-Aware Modeling with Electronic Health Record and Electrocardiogram Data (submitted to IEEE Journal of Biomedical and Health Informatics).

### Supplementary Material
The supplementary materials provide extended tables, figures, and detailed dataset descriptions.
* [Supplementary Document (PDF)](SupplementaryMaterial/SupplementaryMaterial.pdf)
* [Additional Supplementary Files](SupplementaryMaterial/)

### MedM2T Repository Structure
* src/: Implementation of MedM2T framework.
* dataset/: Dataset class definitions and preprocessing scripts for EHR and ECG data.

### Data Availability
This project uses the [MIMIC-IV database (version 2.2)](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV-ECG database (version 1.0)](https://physionet.org/content/mimic-iv-ecg/1.0/) as the primary data source. 

Due to the restricted-access policy, we cannot release any raw data within this repository.  
Accessing MIMIC-IV requires completing the required training and obtaining approval for access through PhysioNet: [PhysioNet Credentialing Process](https://physionet.org/about/database/)

### Preprocessing Pipelines
To ensure reproducibility, we provide source code for preprocessing in the `FirstICU` folder:  
- **Task 2 (In-hospital Mortality Prediction)**  
- **Task 3 (Length of Stay Prediction)**  

These scripts demonstrate the pipeline used to process the raw MIMIC-IV data.  
Users with authorized access to MIMIC-IV can follow these pipelines to reproduce the datasets for training and evaluation.

### Acknowledgment
The ResNet in this project is adapted from [antonior92/automatic-ecg-diagnosis](https://github.com/antonior92/automatic-ecg-diagnosis).  
We have modified and extended the original implementation to fit the MedM2T framework.  
If you use this code, please also cite the original repository/paper as indicated by the authors.  