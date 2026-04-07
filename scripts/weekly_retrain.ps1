# weekly_retrain.ps1
$PROJECT = "C:\Users\arnov\Desktop\energy-demand-forecast"
$PYTHON  = "C:\Users\arnov\AppData\Local\Programs\Python\Python312\python.exe"
$LOG     = "$PROJECT\logs\weekly_retrain.log"

New-Item -ItemType Directory -Force -Path "$PROJECT\logs" | Out-Null
Set-Location $PROJECT

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format yyyy-MM-dd_HH:mm:ss)] $Message"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

Write-Log "=== Weekly retrain started ==="

Write-Log "Step 1/3 - Retraining XGBoost..."
$out = & $PYTHON scripts\train.py --model xgboost 2>&1
$out | Add-Content -Path $LOG
if ($LASTEXITCODE -ne 0) { Write-Log "ERROR: XGBoost failed"; exit 1 }
Write-Log "XGBoost OK"

Write-Log "Step 2/3 - Retraining Ridge..."
$out = & $PYTHON scripts\train.py --model ridge 2>&1
$out | Add-Content -Path $LOG
if ($LASTEXITCODE -ne 0) { Write-Log "ERROR: Ridge failed"; exit 1 }
Write-Log "Ridge OK"

Write-Log "Step 3/3 - Backfilling last 14 days..."
$script = @"
from datetime import date, timedelta
import subprocess, sys, pandas as pd
from pathlib import Path
start = date.today() - timedelta(days=14)
end   = date.today() + timedelta(days=1)
d = start
while d <= end:
    r = subprocess.run([sys.executable, "scripts/infer.py", "--date", d.isoformat()], capture_output=True, text=True)
    ok = "Predictions saved" in r.stdout + r.stderr
    print(f"{d}  OK" if ok else f"{d}  FAIL")
    d += timedelta(days=1)
"@
$out = & $PYTHON -c $script 2>&1
$out | Add-Content -Path $LOG
Write-Log "Backfill OK"

Write-Log "=== Weekly retrain complete ==="
