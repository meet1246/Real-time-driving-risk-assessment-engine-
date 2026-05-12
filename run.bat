@echo off
REM Run from repo folder so risk_engine imports work.
cd /d "%~dp0"
python main.py %*
