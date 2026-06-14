# Run CoppeliaSim headless export of .ttm models listed in export_config.txt
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ExportRoot = Join-Path $RepoRoot "exports\coppelia_edu"
$ConfigPath = Join-Path $ExportRoot "export_config.txt"
$ConfigExample = Join-Path $ExportRoot "export_config.example.txt"
$CoppeliaExe = "C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe"
$AddonLua = Join-Path $PSScriptRoot "coppelia_export_models.lua"

if (-not (Test-Path $ExportRoot)) {
    New-Item -ItemType Directory -Path $ExportRoot -Force | Out-Null
}

if (-not (Test-Path $ConfigPath)) {
    if (Test-Path $ConfigExample) {
        Copy-Item $ConfigExample $ConfigPath
        Write-Host "Created $ConfigPath from example."
    } else {
        throw "Missing export config: $ConfigPath"
    }
}

# Pre-create output subdirs (CoppeliaSim export cannot mkdir)
Get-Content $ConfigPath | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line -notmatch "=") {
        $rel = $line
        if ($rel.StartsWith("urdf:")) {
            $rel = $rel.Substring(5)
        }
        $rel = $rel -replace "\\", "/"
        if ($rel -match "\.ttm$") {
            $pathNoExt = $rel -replace '\.ttm$',''
            $stem = ($pathNoExt -split '/')[-1] -replace ' ', '_'
            $parent = $pathNoExt -replace '/[^/]+$',''
            if ($parent) {
                $outDir = Join-Path $ExportRoot ($parent + "/" + $stem)
            } else {
                $outDir = Join-Path $ExportRoot $stem
            }
            New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        }
    }
}

if (-not (Test-Path $CoppeliaExe)) {
    throw "CoppeliaSim not found: $CoppeliaExe"
}

Write-Host "Starting CoppeliaSim export (headless)..."
Write-Host "Config: $ConfigPath"
Write-Host "Addon:  $AddonLua"

$CoppeliaDir = Split-Path $CoppeliaExe
$StagingConfig = Join-Path $CoppeliaDir "coppelia_export_config.txt"
Copy-Item $ConfigPath $StagingConfig -Force

Push-Location $CoppeliaDir
try {
    & $CoppeliaExe -h -b $AddonLua -q -vinfos
} finally {
    Pop-Location
    Remove-Item $StagingConfig -ErrorAction SilentlyContinue
}

$LogPath = Join-Path $ExportRoot "export_log.txt"
if (Test-Path $LogPath) {
    Write-Host "`n--- export_log.txt ---"
    Get-Content $LogPath
}

Write-Host "`nNext: python robocasa/scripts/asset_scripts/import_coppelia_batch.py"
