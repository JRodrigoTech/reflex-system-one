param(
  [switch] $BannerAlreadyShown,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $ReflexArgs
)

$root = [System.IO.Path]::GetFullPath($PSScriptRoot + "\..")
if (-not $BannerAlreadyShown) {
  $brandingPath = Join-Path $root "assets\reflex-branding.txt"
  if (-not (Test-Path -LiteralPath $brandingPath -PathType Leaf)) {
    throw "The shared Reflex branding resource is missing."
  }
  $brandingLines = [System.IO.File]::ReadAllText($brandingPath, [System.Text.Encoding]::UTF8) -split "\r?\n"
  $taglineMarker = [Array]::IndexOf($brandingLines, "--- REFLEX_TAGLINE ---")
  if ($brandingLines.Length -lt 4 -or $brandingLines[0] -ne "--- REFLEX_WORDMARK ---" -or $taglineMarker -lt 2 -or $taglineMarker -ge ($brandingLines.Length - 1)) {
    throw "The shared Reflex branding resource is invalid."
  }
  Write-Host ($brandingLines[1..($taglineMarker - 1)] -join [System.Environment]::NewLine)
  Write-Host ""
  Write-Host ($brandingLines[($taglineMarker + 1)..($brandingLines.Length - 1)] -join [System.Environment]::NewLine)
  Write-Host ""
}
$env:REFLEX_BANNER_SHOWN = "1"
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $root
$env:REFLEX_ROOT = $root
$env:REFLEX_BOOTSTRAPPED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$runtimeRoot = Join-Path $root "runtime"
$manifestPath = Join-Path $root "manifests\runtime.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$stagingMargin = $manifest.staging_margin_bytes
if ($null -eq $stagingMargin -or $stagingMargin -is [bool] -or [long]$stagingMargin -lt 1) {
  throw "Pinned runtime staging margin is invalid."
}
$pythonSpec = $manifest.python
if ($null -eq $pythonSpec.estimated_unpacked_bytes -or $pythonSpec.estimated_unpacked_bytes -is [bool] -or [long]$pythonSpec.estimated_unpacked_bytes -lt 1) {
  throw "Pinned Python disk estimate is invalid."
}
$pythonHome = Join-Path $runtimeRoot "python"
$pythonExe = Join-Path $pythonHome "python\python.exe"
$envPath = Join-Path $runtimeRoot "envs\core"
$envPython = Join-Path $envPath "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
$pipCachePath = Join-Path $runtimeRoot "cache\pip"
$env:PIP_CACHE_DIR = $pipCachePath
New-Item -ItemType Directory -Force -Path $pipCachePath | Out-Null
$lockPath = Join-Path $runtimeRoot "state\bootstrap.lock"
$stateRoot = Split-Path -Parent $lockPath
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$lockToken = [Guid]::NewGuid().ToString("N")
function Assert-WithinRoot([string] $Candidate, [string] $Parent) {
  $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
  $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
  if (-not $candidateFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "A generated runtime path escaped its managed directory."
  }
}
function Invoke-PendingReinstall {
  $requestPath = Join-Path (Join-Path $runtimeRoot "state") "reinstall.request.json"
  if (-not (Test-Path -LiteralPath $requestPath)) { return $false }
  $helperPath = Join-Path $PSScriptRoot "complete_reinstall.ps1"
  Assert-WithinRoot $requestPath $runtimeRoot
  Assert-WithinRoot $helperPath $root
  & $helperPath -Root $root
  return $true
}
function Get-LockSha256([string] $Path) {
  $value = & $pythonExe -m reflex.lock_hash $Path
  if ($LASTEXITCODE -ne 0 -or -not $value) {
    throw "The dependency lock could not be hashed by the pinned portable Python runtime."
  }
  return ([string]$value).Trim()
}
function Test-LockSha256([string] $Path, [string] $Candidate) {
  $check = "import sys;from pathlib import Path;from reflex.lock_hash import lock_hash_matches;sys.exit(0 if lock_hash_matches(sys.argv[2],Path(sys.argv[1])) else 1)"
  & $pythonExe -c $check $Path $Candidate | Out-Null
  return ($LASTEXITCODE -eq 0)
}
function Assert-MinimumFreeSpace([string] $Path, [long] $RequiredBytes) {
  $driveName = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path)).Substring(0, 1)
  $drive = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
  if ($null -eq $drive -or $null -eq $drive.Free -or [long]$drive.Free -lt $RequiredBytes) {
    throw "The Reflex drive does not have the pinned staging margin required to install safely."
  }
}
function Test-PortablePython([string] $Executable, [string] $ExpectedVersion) {
  if (-not (Test-Path -LiteralPath $Executable)) { return $false }
  try {
    $versionOutput = & $Executable --version 2>&1
    $versionExitCode = $LASTEXITCODE
    if ($versionExitCode -ne 0 -or "$versionOutput".Trim() -ne "Python $ExpectedVersion") { return $false }
    $pipOutput = & $Executable -m pip --version 2>&1
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}
$stageCore = Join-Path (Join-Path $runtimeRoot "envs") "core.new"
$oldCore = Join-Path (Join-Path $runtimeRoot "envs") "core.old"
Assert-WithinRoot $lockPath $runtimeRoot
Assert-WithinRoot $stageCore $runtimeRoot
Assert-WithinRoot $oldCore $runtimeRoot
try {
  $lockData = @{ pid = $PID; process_started = (Get-Process -Id $PID).StartTime.ToUniversalTime().Ticks.ToString(); started = (Get-Date).ToUniversalTime().ToString("o"); token = $lockToken; operation = "bootstrap" } | ConvertTo-Json -Compress
  $stream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($lockData)
  $stream.Write($bytes, 0, $bytes.Length)
  $stream.Dispose()
} catch {
  if (Test-Path -LiteralPath $lockPath) {
    try {
      $oldLock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
      $oldProcess = Get-Process -Id ([int]$oldLock.pid) -ErrorAction SilentlyContinue
      if ($null -ne $oldProcess) {
        $oldStart = $oldProcess.StartTime.ToUniversalTime().Ticks.ToString()
        if ($oldStart -eq [string]$oldLock.process_started) {
          Write-Error "Another Reflex bootstrap operation is active."
          exit 2
        }
      }
      $stale = $lockPath + ".stale-" + [Guid]::NewGuid().ToString("N")
      Assert-WithinRoot $stale $runtimeRoot
      Move-Item -LiteralPath $lockPath -Destination $stale -ErrorAction Stop
      Remove-Item -LiteralPath $stale -Force -ErrorAction Stop
      $stream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
      $bytes = [System.Text.Encoding]::UTF8.GetBytes($lockData)
      $stream.Write($bytes, 0, $bytes.Length)
      $stream.Dispose()
    } catch {
      Write-Error "Bootstrap lock is live or cannot be verified safely; inspect runtime/state/bootstrap.lock."
      exit 2
    }
  } else {
    Write-Error "Another Reflex bootstrap operation is active."
    exit 2
  }
}

try {
  if (Invoke-PendingReinstall) { exit 23 }
  $pythonReady = Test-PortablePython $pythonExe $pythonSpec.version
  if (-not $pythonReady) {
    $pythonRequiredSpace = [long]$manifest.staging_margin_bytes + 2 * [long]$pythonSpec.estimated_unpacked_bytes
    Assert-MinimumFreeSpace $runtimeRoot $pythonRequiredSpace
    $archive = Join-Path $runtimeRoot "cache\python-standalone.tar.gz.partial"
    $cache = Split-Path -Parent $archive
    $stage = Join-Path $runtimeRoot "python.new"
    $old = Join-Path $runtimeRoot "python.old"
    Assert-WithinRoot $archive $runtimeRoot
    Assert-WithinRoot $stage $runtimeRoot
    Assert-WithinRoot $old $runtimeRoot
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    try {
      $downloaded = $false
      for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
          Invoke-WebRequest -Uri $pythonSpec.archive -OutFile $archive -TimeoutSec 120
          $downloaded = $true
          break
        } catch {
          if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue }
          if ($attempt -eq 3) { throw }
          Start-Sleep -Seconds $attempt
        }
      }
      if (-not $downloaded) { throw "Portable Python download failed." }
      $hashStream = [System.IO.File]::OpenRead($archive)
      try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
          $actualHash = [System.BitConverter]::ToString($sha.ComputeHash($hashStream)).Replace("-", "").ToLowerInvariant()
        } finally {
          $sha.Dispose()
        }
      } finally {
        $hashStream.Dispose()
      }
      if ($actualHash -ne $pythonSpec.sha256.ToLowerInvariant()) {
        throw "Portable Python SHA-256 verification failed. Staged runtime was not promoted."
      }
      if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
      New-Item -ItemType Directory -Force -Path $stage | Out-Null
      Push-Location -LiteralPath $stage
      try {
        # Pass a relative ASCII archive path to tar.exe. Windows PowerShell 5.1
        # can mis-encode non-ASCII absolute paths in native command arguments.
        & tar.exe -xzf "..\cache\python-standalone.tar.gz.partial"
        $extractExitCode = $LASTEXITCODE
      } finally {
        Pop-Location
      }
      if ($extractExitCode -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $stage "python\python.exe"))) {
        throw "Portable Python archive could not be extracted and verified."
      }
      $candidate = Join-Path $stage "python\python.exe"
      if (-not (Test-PortablePython $candidate $pythonSpec.version)) {
        throw "Portable Python version or pip did not match the runtime manifest."
      }
      if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force }
      if (Test-Path -LiteralPath $pythonHome) { Move-Item -LiteralPath $pythonHome -Destination $old }
      try {
        Move-Item -LiteralPath $stage -Destination $pythonHome
      } catch {
        if ((Test-Path -LiteralPath $old) -and -not (Test-Path -LiteralPath $pythonHome)) { Move-Item -LiteralPath $old -Destination $pythonHome }
        throw
      }
      if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force }
      Remove-Item -LiteralPath $archive -Force
    } catch {
      if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
      if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue }
      throw
    }
  }

  $coreLock = Join-Path $root "requirements\core.lock"
  $coreLockHash = Get-LockSha256 $coreLock
  if ($coreLockHash -ne [string]$manifest.environments.core.sha256) {
    throw "Core dependency lock does not match the pinned runtime manifest."
  }
  $coreMarker = Join-Path $envPath "reflex-lock.json"
  $coreHealthy = $false
  if ((Test-Path -LiteralPath $envPython) -and (Test-Path -LiteralPath $coreMarker)) {
    try {
      $marker = Get-Content -LiteralPath $coreMarker -Raw | ConvertFrom-Json
      if (Test-LockSha256 $coreLock ([string]$marker.sha256)) {
        & $envPython -c "import sys;sys.path=[entry for entry in sys.path if entry];import fastapi,uvicorn,huggingface_hub,reflex,pathlib,os; expected=(pathlib.Path(os.environ['REFLEX_ROOT'])/'reflex'/'__init__.py').resolve(); actual=pathlib.Path(reflex.__file__).resolve(); assert actual == expected"
        $coreHealthy = ($LASTEXITCODE -eq 0)
        if ($coreHealthy) {
          & $envPython -m pip check | Out-Null
          $coreHealthy = ($LASTEXITCODE -eq 0)
        }
      }
    } catch { $coreHealthy = $false }
  }
  if (-not $coreHealthy) {
    $coreRequiredSpace = [long]$manifest.staging_margin_bytes + 2 * [long]$manifest.environments.core.estimated_unpacked_bytes
    Assert-MinimumFreeSpace $runtimeRoot $coreRequiredSpace
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $envPath) | Out-Null
    if (Test-Path -LiteralPath $stageCore) { Remove-Item -LiteralPath $stageCore -Recurse -Force }
    & ".\runtime\python\python\python.exe" -m venv ".\runtime\envs\core.new"
    if ($LASTEXITCODE -ne 0) { throw "Core Python environment creation failed." }
    $stagePython = Join-Path $stageCore "Scripts\python.exe"
    & $stagePython -m pip install --disable-pip-version-check --no-warn-script-location --require-hashes -r .\requirements\core.lock
    if ($LASTEXITCODE -ne 0) { throw "Core Reflex dependencies could not be installed." }
    & $stagePython -m pip install --disable-pip-version-check --no-warn-script-location --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw "Reflex package installation failed." }
    & $stagePython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Core Reflex dependencies failed consistency checks." }
    $markerPath = Join-Path $stageCore "reflex-lock.json"
    $markerJson = @{ sha256 = $coreLockHash; family = "core" } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($markerPath, $markerJson + "`n", [System.Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $oldCore) { Remove-Item -LiteralPath $oldCore -Recurse -Force }
    if (Test-Path -LiteralPath $envPath) { Move-Item -LiteralPath $envPath -Destination $oldCore }
    try {
      Move-Item -LiteralPath $stageCore -Destination $envPath
    } catch {
      if ((Test-Path -LiteralPath $oldCore) -and -not (Test-Path -LiteralPath $envPath)) { Move-Item -LiteralPath $oldCore -Destination $envPath }
      throw
    }
    if (Test-Path -LiteralPath $oldCore) { Remove-Item -LiteralPath $oldCore -Recurse -Force }
  }
  & $envPython -m reflex @ReflexArgs
  $reflexExitCode = $LASTEXITCODE
  if (Invoke-PendingReinstall) { exit 23 }
  exit $reflexExitCode
} finally {
  try {
    $held = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
    if ($held.token -eq $lockToken) { Remove-Item -LiteralPath $lockPath -Force }
  } catch { }
  Pop-Location
}
