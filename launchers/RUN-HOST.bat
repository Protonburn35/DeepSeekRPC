@echo off
setlocal
"%~dp0..\runtime\Scripts\python.exe" -m deepseekrpc.cli serve --stage A %*
exit /b %errorlevel%
