@echo off
setlocal
chcp 65001 >nul 2>&1
set "REFLEX_ROOT=%~dp0"
set "REFLEX_BANNER_SHOWN="
:bootstrap
if defined REFLEX_BANNER_SHOWN (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" -BannerAlreadyShown %*
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
)
set "BOOTSTRAP_EXIT=%ERRORLEVEL%"
if "%BOOTSTRAP_EXIT%"=="23" (
  set "REFLEX_BANNER_SHOWN=1"
  goto bootstrap
)
exit /b %BOOTSTRAP_EXIT%
