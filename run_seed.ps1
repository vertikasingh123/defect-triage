# Run the full stage-2 experiment for one seed:  .\run_seed.ps1 1
param([Parameter(Mandatory=$true)][int]$Seed)
$ErrorActionPreference = "Stop"
python scripts/2_make_splits_and_synthetic.py --seed $Seed
python scripts/3_refine_with_diffusion.py --seed $Seed
python scripts/5_train_classifier.py --regime all --seed $Seed --input full
python scripts/5_train_classifier.py --regime all --seed $Seed --input crop
python scripts/5_train_classifier.py --regime all --seed $Seed --input oracle
python scripts/6_evaluate.py --input crop
