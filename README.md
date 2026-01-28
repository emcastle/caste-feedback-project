
# Caste_Project — Setup & Run (Windows / Conda)

This project uses a **Conda environment** (not `python -m venv`) and is designed so a user can run everything by double-clicking a `.bat` file.

> In Census enterprise environments, Conda is typically configured to use the internal Nexus proxy. This repo’s setup assumes Conda is already installed and configured on the machine (e.g., via Software Center with configurations set).

---
## Notes to Self: 
These are ideas that still need to be developed
1. temporal layer 
- look at sentiment per produc tover time (not sure what the cutoff ranges will be)
2. normalize by organization/person 
- enforce a sort of de-duplication/weighting/concentration monitoring to not skew 
  feedback of a product 
3. track responses and monitor 
4. actionable summary per product
- may also break it down by year or other meaningful variables 

---

## Repo Files You’ll Use

- `environment.yml` — the single source of truth for dependencies
- `run_conda.bat` — creates/updates the Conda env and runs the entrypoint script
- `sanity_check.py` — temporary entrypoint used until a real pipeline exists (verifies imports)

---

## 1) How to Start (First Run)

### Option A (recommended): double-click
1. Open the project folder in File Explorer
2. Double-click: `run_conda.bat`

This will:
- create the Conda environment (first time only)
- install/update dependencies from `environment.yml`
- run the entrypoint (currently `scripts/sanity_check.py`)

### Option B: run from terminal (PowerShell, CMD, or Anaconda Prompt)
From the project root:
```powershell
.\run_conda.bat
```

OR from VS Code use the same 
>> .\run_conda.bat

then to activate the conda venv: "conda activate feedback"
the next prompt should include "(feedback)" set as the venv


## 2) How to Update Dependencies
a. Edit the environment.yml and add/remove packages under dependencies
b. To apply the update (without running the pipeline or rebuilding the venv)
>> .\run_conda.yaml install

## )3 How the Environment Works
The Conda environment is set in the environment.yml
"name: feedback"

**To Activate the Environment**
- ensure there isn't another conda active first: 
``` 
# first 
conda deactivate

# second 
conda activate feedback

# third, run a script
python script_name.py
```

---

## FILE ORGANIZATION

caste-feedback-project/
├─ pyproject.toml
├─ environment.yml
├─ README.md
├─ run_extraction_smoke_test.py
├─ run_conda.bat
├─ sanity_check.py
├─ data/
│  ├─ _test_input/
│  │  ├─ pdf/
│  │  ├─ docx/
│  │  ├─ csv/
│  │  ├─ pptx/
│  │  └─ json/
│  ├─ _test_output/          # ingestion outputs (parquet)
│  └─ _seg_output/           # segmentation outputs (parquet)
├─ notebooks/
└─ src/
   └─ Caste_Project/
      ├─ __init__.py
      ├─ ingest/
      │  ├─ __init__.py
      │  ├─ discover.py
      │  └─ handlers/
      │     ├─ __init__.py
      │     ├─ document_pdf.py
      │     ├─ document_docx.py
      │     ├─ presentation_pptx.py
      │     ├─ tabular_csv.py
      │     ├─ tabular_excel.py
      │     ├─ structured_json.py
      │     └─ document_txt.py        # even if no txt exists now 
      ├─ segment/
      │  ├─ __init__.py
      │  ├─ segment_pdf.py
      │  ├─ segment_docx.py
      │  ├─ segment_json.py
      │  ├─ segment_csv.py
      │  ├─ segment_excel.py
      │  └─ segment_pptx.py
      ├─ parse/                       # Work In Progess
      │  ├─ __init__.py
      │  └─ (future: parse_pdf.py, parse_docx.py, etc.)
      └─ utils/                    
         ├─ __init__.py
         └─ text.py





