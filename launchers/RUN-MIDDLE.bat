@echo off
setlocal
if not exist "%~dp0..\.install-complete.json" (
  echo Run Install.ps1 successfully before launching this bundle.
  exit /b 2
)
"%~dp0..\runtime\Scripts\python.exe" -I -m deepseekrpc.bundle_launcher --bundle "%~dp0.." --stage B %*
exit /b %errorlevel%
