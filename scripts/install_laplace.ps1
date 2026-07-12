param(
    [switch]$SkipPathUpdate
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$Venv = Join-Path $Repo ".venv"
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $Python) { throw "Python 3.11+ is required. Install Python or use the repository .venv." }

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $Python.Source -m venv $Venv
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Spec = "$Repo[dev]"
try {
    & $VenvPython -m pip install -e $Spec
    if ($LASTEXITCODE -ne 0) { throw "pip exited with code $LASTEXITCODE" }
} catch {
    Write-Warning "Editable pip installation could not complete (often because build dependencies are unavailable offline). Using a local source-path fallback for this launcher."
    $SitePackages = (& $VenvPython -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
    $Pth = Join-Path $SitePackages "laplace_local_source.pth"
    [IO.File]::WriteAllText($Pth, (Join-Path $Repo "src"), [Text.Encoding]::UTF8)
}

$LauncherDir = Join-Path $HOME "AppData\Local\Programs\Laplace"
$Launcher = Join-Path $LauncherDir "laplace.cmd"
New-Item -ItemType Directory -Force -Path $LauncherDir | Out-Null
$LauncherBody = "@echo off`r`n`"$VenvPython`" -m research_workspace.laplace_cli %*`r`n"
[IO.File]::WriteAllText($Launcher, $LauncherBody, [Text.Encoding]::ASCII)

if (-not $SkipPathUpdate) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $Entries = @($UserPath -split ";" | Where-Object { $_ })
    if ($Entries -notcontains $LauncherDir) {
        [Environment]::SetEnvironmentVariable("Path", (($Entries + $LauncherDir) -join ";"), "User")
    }
}

Write-Output "Laplace installed: $Launcher"
Write-Output "Open a new terminal to use: laplace --version"
