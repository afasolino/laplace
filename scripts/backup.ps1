param([string]$Destination = 'backup')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Copy-Item -Recurse -Force data, configs, schemas, prompts, outputs, $Destination

