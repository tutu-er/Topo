$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed. Inspect build/main.log.' }
    $logText = Get-Content -LiteralPath 'build\main.log' -Raw
    if ($logText -match 'There were undefined references|There were undefined citations|Missing character:|Overfull') {
        throw 'Unresolved references, missing glyphs or overflowing content remain. Inspect build/main.log.'
    }
    Write-Output ('PDF: ' + (Join-Path $PSScriptRoot 'build\main.pdf'))
} finally {
    Pop-Location
}
