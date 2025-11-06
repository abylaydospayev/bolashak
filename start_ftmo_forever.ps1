# Auto-Start FTMO Trading Bot
# This script runs on VM startup and keeps the bot running forever

$LogFile = "C:\bolashak\live_trading\logs\autostart.log"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}

Write-Log "========================================="
Write-Log "FTMO Trading Bot - Auto-Start Script"
Write-Log "========================================="

# Ensure logs directory exists
New-Item -Path "C:\bolashak\live_trading\logs" -ItemType Directory -Force | Out-Null

# Wait for network to be available
Write-Log "Waiting for network..."
Start-Sleep -Seconds 30

# Activate virtual environment and run bot forever
cd C:\bolashak

while ($true) {
    Write-Log "Starting FTMO trading bot..."
    
    try {
        # Activate venv and run bot
        & .\.venv\Scripts\python.exe live_trading\demo_bot.py
        
        $exitCode = $LASTEXITCODE
        Write-Log "Bot stopped with exit code: $exitCode"
        
    } catch {
        Write-Log "ERROR: $($_.Exception.Message)"
    }
    
    Write-Log "Restarting bot in 30 seconds..."
    Start-Sleep -Seconds 30
}
