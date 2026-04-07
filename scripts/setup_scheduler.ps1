# ============================================================
# setup_scheduler.ps1
# Run once (as Administrator) to register the two scheduled tasks.
#
# Usage:
#   Right-click PowerShell -> "Run as Administrator"
#   cd C:\Users\arnov\Desktop\energy-demand-forecast
#   .\scripts\setup_scheduler.ps1
# ============================================================

$PROJECT = "C:\Users\arnov\Desktop\energy-demand-forecast"

# ── Task 1: Daily pipeline at 14:00 ─────────────────────────
$dailyAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$PROJECT\scripts\daily_pipeline.ps1`"" `
    -WorkingDirectory $PROJECT

$dailyTrigger = New-ScheduledTaskTrigger -Daily -At "14:00"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable   # Run if PC was off at scheduled time

Register-ScheduledTask `
    -TaskName "EnergyForecast_DailyPipeline" `
    -TaskPath "\EnergyForecast\" `
    -Action $dailyAction `
    -Trigger $dailyTrigger `
    -Settings $settings `
    -Description "Daily: update data + rebuild dataset + J+1 inference (14:00)" `
    -RunLevel Highest `
    -Force

Write-Host "✅ Daily task registered: EnergyForecast_DailyPipeline (14:00 every day)"

# ── Task 2: Weekly retrain every Monday at 02:00 ─────────────
$weeklyAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$PROJECT\scripts\weekly_retrain.ps1`"" `
    -WorkingDirectory $PROJECT

$weeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "02:00"

$weeklySettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 60) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName "EnergyForecast_WeeklyRetrain" `
    -TaskPath "\EnergyForecast\" `
    -Action $weeklyAction `
    -Trigger $weeklyTrigger `
    -Settings $weeklySettings `
    -Description "Weekly: retrain XGBoost + Ridge (Monday 02:00)" `
    -RunLevel Highest `
    -Force

Write-Host "✅ Weekly task registered: EnergyForecast_WeeklyRetrain (Monday 02:00)"

Write-Host ""
Write-Host "Tasks visible in: Task Scheduler > Task Scheduler Library > EnergyForecast"
Write-Host "To remove: Unregister-ScheduledTask -TaskName 'EnergyForecast_DailyPipeline' -Confirm:`$false"
