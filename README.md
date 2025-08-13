# README

Submission for APSIPA ASC 2025 GC. The checkpoint is in **"./log/APSIPA_2025_ASC_GC/best_model_2"**.

Checkpoint: https://drive.google.com/file/d/1NauKrMAkjf6MebvAGxYqUOKajf1uYE5w/view?usp=sharing

## Inference

### Step 1: Python Running Environment
```shell
conda create -n ASC python=3.10
conda activate ASC
pip install -r requirement.txt
```

### Step 2: Pre-Processing of the Evaluation Data

Down-sample the evaluation data to 44.1 kHz and move them as **"./ICME2024_GC_ASC_eval/*.wav"**.

Put the meta data of evaluation dataset as **"./metadata/APSIPA2025_GC_ASC_eval_metadata.csv"**.

### Step 3: Feature extraction

```shell
# Feature extraction of evaluation dataset:
python3 feature_extraction.py --dataset eval
```

The features will be saved at **"./feature/eval"**.

### Step4: Evaluate Model

```shell
# Model testing, output predicted results of evaluation dataset.
python test.py
```

The results will be saved as **"./log/APSIPA_2025_ASC_GC/eval_results.csv"**.

