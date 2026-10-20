@echo off
setlocal
"%~dp0..\runtime\Scripts\python.exe" -m deepseekrpc.cli serve %*
exit /b %errorlevel%
