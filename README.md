
# Caste_Project — Setup & Run (Windows / Conda)

This project uses a **Conda environment** (not `python -m venv`) and is designed so a user can run everything by double-clicking a `.bat` file.

> In Census enterprise environments, Conda is typically configured to use the internal Nexus proxy. This repo’s setup assumes Conda is already installed and configured on the machine (e.g., via Software Center with configurations set).

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


