@echo off
rem Start Welder in safe mode: no splash screen, no floating panel windows, the default layout
rem (the saved layout is kept as layout.ini.bak). Use this if Welder opens to a blank window or
rem closes right after the splash screen.
cd /d "%~dp0"
set OFX_PLUGIN_PATH=%CD%\ofx-test-plugins;%OFX_PLUGIN_PATH%
start "" "%~dp0bin\welder_ui.exe" --safe-mode %*
