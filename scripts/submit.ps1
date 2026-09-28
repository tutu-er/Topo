param(
    [string] $Message,
    [switch] $Push,
    [switch] $Preview,
    [string[]] $Paths
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Git {
    param([string[]] $GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE)."
    }
}

Push-Location $projectRoot
try {
    $root = (& git rev-parse --show-toplevel).Trim()
    if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($root) -ne [IO.Path]::GetFullPath($projectRoot)) {
        throw 'Run this script from the Topo Git checkout.'
    }

    $remote = (& git remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or $remote -notmatch '^(https://github\.com/|git@github\.com:)tutu-er/Topo(\.git)?$') {
        throw "Unexpected origin: $remote"
    }

    $branch = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) {
        throw 'Check out a branch before submitting.'
    }

    $name = (& git config user.name)
    $email = (& git config user.email)
    if ([string]::IsNullOrWhiteSpace($name) -or [string]::IsNullOrWhiteSpace($email)) {
        throw 'Set Git user.name and user.email before submitting.'
    }

    Write-Host "Branch: $branch"
    Write-Host "Author: $name <$email>"
    Write-Host "Origin: $remote"

    if ($Preview) {
        if ($Paths.Count -gt 0) {
            Invoke-Git -GitArgs (@('status', '--short', '--') + $Paths)
        }
        else {
            Invoke-Git -GitArgs @('status', '--short')
        }
        return
    }
    if ([string]::IsNullOrWhiteSpace($Message)) {
        throw 'Provide a commit message with -Message.'
    }

    if ($Paths.Count -gt 0) {
        & git diff --cached --quiet
        if ($LASTEXITCODE -eq 1) {
            throw 'Scoped submission requires an empty staging area. Commit or unstage existing staged changes first.'
        }
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not inspect the staging area.'
        }
        Invoke-Git -GitArgs (@('add', '-A', '--') + $Paths)
    }
    else {
        Invoke-Git -GitArgs @('add', '-A')
    }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'Nothing to commit.'
        return
    }
    if ($LASTEXITCODE -ne 1) {
        throw 'Could not inspect staged changes.'
    }

    $paths = @(& git -c core.quotePath=false diff --cached --name-only --diff-filter=ACMRT)
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not list staged files.'
    }
    foreach ($path in $paths) {
        $filename = [IO.Path]::GetFileName($path)
        if ($filename -match '^(\.env(\..*)?|id_(rsa|ed25519)|.*\.(pem|p12|pfx|key)|credentials(\..*)?)$') {
            throw "Possible credential file staged: $path"
        }
        if ((Test-Path -LiteralPath $path -PathType Leaf) -and (Get-Item -LiteralPath $path).Length -gt 25MB) {
            throw "File exceeds the 25 MB submission limit: $path"
        }
    }

    $secretFiles = @(& git grep --cached -I -l -E 'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----' -- .)
    if ($LASTEXITCODE -gt 1) {
        throw 'Credential scan failed.'
    }
    if ($secretFiles.Count -gt 0) {
        throw "Possible credential text in staged files: $($secretFiles -join ', ')"
    }

    Write-Host "Staged files: $($paths.Count)"
    Invoke-Git -GitArgs @('diff', '--cached', '--stat')
    Invoke-Git -GitArgs @('commit', '-m', $Message)
    if ($Push) {
        Invoke-Git -GitArgs @('push', '-u', 'origin', $branch)
    }
    else {
        Write-Host 'Committed locally. Add -Push to submit to GitHub.'
    }
}
finally {
    Pop-Location
}
