param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PytestArgs
)

$projectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-TopoPython {
    if ($env:CONDA_DEFAULT_ENV -eq "Topo" -and $env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        $resolved = & conda.exe run -n Topo python -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $pythonPath = ($resolved | Select-Object -Last 1).Trim()
            if (Test-Path -LiteralPath $pythonPath) {
                return $pythonPath
            }
        }
    }

    $fallback = "D:\apps\miniconda3\envs\Topo\python.exe"
    if (Test-Path -LiteralPath $fallback) {
        return $fallback
    }

    throw "Conda environment 'Topo' was not found. Create it or run: conda activate Topo"
}

$python = Resolve-TopoPython
$previousPythonPath = $env:PYTHONPATH
try {
    if ([string]::IsNullOrEmpty($previousPythonPath)) {
        $env:PYTHONPATH = $projectRoot
    }
    else {
        $env:PYTHONPATH = "$projectRoot$([IO.Path]::PathSeparator)$previousPythonPath"
    }
    & $python -m pytest @PytestArgs
    exit $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
