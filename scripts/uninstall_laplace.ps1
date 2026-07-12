param(
    [switch]$KeepPathEntry
)

$ErrorActionPreference = "Stop"
$LauncherDir = Join-Path $HOME "AppData\Local\Programs\Laplace"
$Launcher = Join-Path $LauncherDir "laplace.cmd"
if (Test-Path -LiteralPath $Launcher) { Remove-Item -LiteralPath $Launcher -Force }
if (-not $KeepPathEntry) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath) {
        $Entries = @($UserPath -split ";" | Where-Object { $_ -and $_ -ne $LauncherDir })
        [Environment]::SetEnvironmentVariable("Path", ($Entries -join ";"), "User")
    }
}
Write-Output "Laplace launcher removed. Repository files, projects, models, and source documents were not deleted."
