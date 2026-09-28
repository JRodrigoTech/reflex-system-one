param(
  [Parameter(Mandatory = $true)]
  [string] $Root
)

$ErrorActionPreference = "Stop"
$rootFull = [System.IO.Path]::GetFullPath($Root)
$runtimeRoot = Join-Path $rootFull "runtime"
$stateRoot = Join-Path $runtimeRoot "state"
$requestPath = Join-Path $stateRoot "reinstall.request.json"

function Assert-WithinRoot([string] $Candidate, [string] $Parent) {
  $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
  $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
  if (-not $candidateFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "A deferred reinstall path escaped its managed directory."
  }
}

function Remove-ManagedTree([string] $Candidate, [string] $Parent) {
  Assert-WithinRoot $Candidate $Parent
  $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\')
  $cursor = [System.IO.Path]::GetFullPath($Candidate)
  while ($true) {
    if (Test-Path -LiteralPath $cursor) {
      $entry = Get-Item -LiteralPath $cursor -Force
      if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Deferred reinstall will not remove a managed path that contains a junction or symbolic link."
      }
    }
    if ($cursor.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) { break }
    if (-not $cursor.StartsWith($parentFull + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
      throw "A deferred reinstall path escaped its managed directory."
    }
    $cursor = [System.IO.Path]::GetDirectoryName($cursor)
  }
  if (Test-Path -LiteralPath $Candidate) {
    Remove-Item -LiteralPath $Candidate -Recurse -Force
  }
}

function Write-StagedUtf8([string] $Path, [string] $Content) {
  Assert-WithinRoot $Path $runtimeRoot
  [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Promote-StagedFile([string] $Staged, [string] $Destination) {
  Assert-WithinRoot $Staged $runtimeRoot
  Assert-WithinRoot $Destination $runtimeRoot
  if (Test-Path -LiteralPath $Destination) {
    [System.IO.File]::Replace($Staged, $Destination, $null)
  } else {
    [System.IO.File]::Move($Staged, $Destination)
  }
}

if (-not (Test-Path -LiteralPath $requestPath)) { return }
Assert-WithinRoot $requestPath $runtimeRoot
$request = Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
if ($request.schema_version -ne 1 -or $request.scope -notin @("runtime", "reset-settings", "full", "family:laya", "family:decider", "family:decision-cuda")) {
  throw "Deferred reinstall request is invalid; no runtime or model data was removed."
}
$expectedConfirmation = if ($request.scope -eq "full") { "DELETE ALL MODELS" } else { "REINSTALL" }
if ($request.confirmation -cne $expectedConfirmation) {
  throw "Deferred reinstall confirmation is invalid; no runtime or model data was removed."
}

$configStage = Join-Path $stateRoot "config.json.reinstall.new"
$stateStage = Join-Path $stateRoot "state.json.reinstall.new"
if ($request.scope -in @("reset-settings", "full")) {
  foreach ($stage in @($configStage, $stateStage)) {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Force }
  }
  $defaultConfig = [ordered]@{
    schema_version = 1
    active_model = "laya"
    laya_variant = "multilingual"
    host = "127.0.0.1"
    port = 1919
    api_key = $null
    log_level = "NORMAL"
    admission_capacity = 16
    max_body_bytes = 1048576
  }
  $defaultState = [ordered]@{ schema_version = 1; setup_complete = $false }
  Write-StagedUtf8 $configStage (($defaultConfig | ConvertTo-Json -Depth 8) + "`n")
  Write-StagedUtf8 $stateStage (($defaultState | ConvertTo-Json -Depth 8) + "`n")
}

if ($request.scope -like "family:*") {
  $family = $request.scope.Substring("family:".Length)
  Remove-ManagedTree (Join-Path (Join-Path $runtimeRoot "envs") $family) $runtimeRoot
} else {
  foreach ($relative in @("python", "python.new", "python.old", "envs", "wsl", "cache", "logs")) {
    Remove-ManagedTree (Join-Path $runtimeRoot $relative) $runtimeRoot
  }
  if ($request.scope -eq "full") {
    Remove-ManagedTree (Join-Path $rootFull "models") $rootFull
  }
}

if ($request.scope -in @("reset-settings", "full")) {
  $preserve = @("bootstrap.lock", "reinstall.request.json", "config.json.reinstall.new", "state.json.reinstall.new")
  foreach ($item in @(Get-ChildItem -LiteralPath $stateRoot -Force)) {
    if ($item.Name -notin $preserve) {
      Remove-Item -LiteralPath $item.FullName -Recurse -Force
    }
  }
  Promote-StagedFile $configStage (Join-Path $stateRoot "config.json")
  Promote-StagedFile $stateStage (Join-Path $stateRoot "state.json")
}

Remove-Item -LiteralPath $requestPath -Force
Write-Output "Deferred reinstall completed; restarting the Reflex bootstrap."
