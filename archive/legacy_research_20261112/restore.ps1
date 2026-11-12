param(
    [switch] $Apply,
    [switch] $IncludeLocal,
    [string] $Only = ''
)
$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$archivePrefix = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'legacy')) + [IO.Path]::DirectorySeparatorChar
$repositoryPrefix = $repository.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json
$rows = @($manifest.tracked_files)
if ($IncludeLocal) {
    $rows = @(Get-Content -LiteralPath (Join-Path $PSScriptRoot 'LOCAL_MANIFEST.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
}
$actions = @()
$unchanged = 0
foreach ($row in $rows) {
    if ($Only -and $row.original -ne $Only -and -not $row.original.StartsWith($Only.TrimEnd('/') + '/')) { continue }
    $source = [IO.Path]::GetFullPath((Join-Path $repository $row.archived))
    $target = [IO.Path]::GetFullPath((Join-Path $repository $row.original))
    if (-not $source.StartsWith($archivePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $target.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Restore path escapes the repository or archive.'
    }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
        throw "Archived source hash changed: $($row.original)"
    }
    if (Test-Path -LiteralPath $target) {
        if ((Get-Item -LiteralPath $target).PSIsContainer -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
            throw "Current file differs; preserve it and select a narrower -Only path: $($row.original)"
        }
        $unchanged++
    } else {
        $actions += [PSCustomObject]@{ Source = $source; Target = $target }
    }
}
if ($Apply) {
    foreach ($item in $actions) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $item.Target) -Force | Out-Null
        Copy-Item -LiteralPath $item.Source -Destination $item.Target
    }
}
[PSCustomObject]@{ Mode = $(if ($Apply) { 'copied_without_overwriting' } else { 'dry_run' }); Files = $actions.Count; ExistingIdentical = $unchanged }
