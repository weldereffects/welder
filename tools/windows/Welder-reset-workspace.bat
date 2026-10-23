@echo off
rem Start Welder with the default workspace (the saved layout is kept as layout.ini.bak).
rem Use this if the panels got into a state you can't fix with Workspace > Reset Workspace.
cd /d "%~dp0"
set OFX_PLUGIN_PATH=%CD%\ofx-test-plugins;%OFX_PLUGIN_PATH%
start "" "%~dp0bin\welder_ui.exe" --reset-workspace %*
