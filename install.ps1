cd resources/python/venv-3.6.8

$CurrentPath = Get-Location

$config = @"
home = $CurrentPath\Python36
implementation = CPython
version_info = 3.6.8.final.0
virtualenv = 20.13.0
include-system-site-packages = false
base-prefix = $CurrentPath\Python36
base-exec-prefix = $CurrentPath\Python36
base-executable = $CurrentPath\Python36\python.exe
"@

Set-Content -Path "pyvenv.cfg" -Value $config

cd ../venv

$CurrentPath = Get-Location

$config = @"
home = $CurrentPath\Python311
implementation = CPython
version_info = 3.11.7.final.0
virtualenv = 20.24.5
include-system-site-packages = false
base-prefix = $CurrentPath\Python311
base-exec-prefix = $CurrentPath\Python311
base-executable = $CurrentPath\Python311\python.exe
"@

Set-Content -Path "pyvenv.cfg" -Value $config

cd ../../..

Write-Output "Venv installation finished"