$ErrorActionPreference = 'Stop'

if (Get-Process AfterFX -ErrorAction SilentlyContinue) {
    throw 'Close After Effects before installing 3DScener.'
}

$extensionRoot = Join-Path $env:APPDATA 'Adobe\CEP\extensions\3DScener'
$files = @(
    'CSXS\manifest.xml',
    'host\scene.jsx',
    'worker\scener.py',
    'worker\models.py',
    'worker\models.json',
    'worker\runtime.py',
    'worker\inference.py',
    'worker\image_input.py',
    'worker\lama_graph.py',
    'worker\scene.py',
    'worker\segmentation.py',
    'worker\mobile_sam.py',
    'worker\requirements.txt',
    'index.html',
    'panel.css',
    'local-runtime.js',
    'panel.js'
)

foreach ($relativePath in $files) {
    $destination = Join-Path $extensionRoot $relativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $relativePath) -Destination $destination -Force
}

# This development build is unsigned. Enable CEP 12 developer extensions for this user.
$cepRegistry = 'HKCU:\Software\Adobe\CSXS.12'
if (-not (Test-Path -LiteralPath $cepRegistry)) {
    New-Item -Path $cepRegistry -Force | Out-Null
}
New-ItemProperty -Path $cepRegistry -Name PlayerDebugMode -PropertyType String -Value '1' -Force | Out-Null

Write-Host "Installed 3DScener at $extensionRoot"
Write-Host 'Open After Effects and choose Window > Extensions > 3DScener.'
