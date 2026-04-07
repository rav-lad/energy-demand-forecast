# daily_pipeline.ps1
$PROJECT = "C:\Users\arnov\Desktop\energy-demand-forecast"
$PYTHON  = "C:\Users\arnov\AppData\Local\Programs\Python\Python312\python.exe"
$LOG     = "$PROJECT\logs\daily_pipeline.log"

New-Item -ItemType Directory -Force -Path "$PROJECT\logs" | Out-Null
Set-Location $PROJECT

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format yyyy-MM-dd_HH:mm:ss)] $Message"
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

Write-Log "=== Daily pipeline started ==="

Write-Log "Step 1/3 - Updating data..."
$out = & $PYTHON scripts\update_data.py 2>&1
$out | Add-Content -Path $LOG
if ($LASTEXITCODE -ne 0) { Write-Log "ERROR: update_data.py failed"; exit 1 }
Write-Log "Data update OK"

Write-Log "Step 2/3 - Rebuilding dataset..."
$out = & $PYTHON scripts\build_dataset.py 2>&1
$out | Add-Content -Path $LOG
if ($LASTEXITCODE -ne 0) { Write-Log "ERROR: build_dataset.py failed"; exit 1 }
Write-Log "Dataset OK"

Write-Log "Step 3/3 - J+1 inference..."
$out = & $PYTHON scripts\infer.py 2>&1
$out | Add-Content -Path $LOG
if ($LASTEXITCODE -ne 0) { Write-Log "ERROR: infer.py failed"; exit 1 }
Write-Log "Inference OK"

Write-Log "=== Daily pipeline complete ==="
