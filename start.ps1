$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = if (Test-Path '.venv/Scripts/python.exe') { (Resolve-Path '.venv/Scripts/python.exe').Path } else { (Get-Command python).Source }
foreach ($port in @(8000, 8501)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "Port $port is already in use." }
}
$apiProcess = $null
$dashboardProcess = $null
try {
    $apiProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m','uvicorn','api.main:app','--host','127.0.0.1','--port','8000') -WindowStyle Hidden -PassThru
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($apiProcess.HasExited) { throw 'API stopped before becoming ready.' }
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2
            if ($health.ready) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw 'API is not ready. Train a compatible model first: python src/train.py' }
    $env:API_URL = 'http://127.0.0.1:8000'
    $dashboardProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m','streamlit','run','dashboard.py','--server.address=127.0.0.1','--server.port=8501','--server.headless=true') -WindowStyle Hidden -PassThru
    Write-Host 'Dashboard: http://127.0.0.1:8501 - press Ctrl+C to stop.'
    while (-not $apiProcess.HasExited -and -not $dashboardProcess.HasExited) { Start-Sleep -Seconds 1 }
} finally {
    foreach ($ownedProcess in @($dashboardProcess, $apiProcess)) {
        if ($null -ne $ownedProcess -and -not $ownedProcess.HasExited) { Stop-Process -Id $ownedProcess.Id }
    }
}
