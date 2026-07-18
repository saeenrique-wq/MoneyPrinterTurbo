Set-Location "C:\Users\saems\MoneyPrinterTurbo"
& .\webui.bat 2>&1 | Out-File -FilePath "webui_log.txt" -Encoding utf8 -Append
