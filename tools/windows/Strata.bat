@echo off
rem Launch the Strata editor. Drag a .strata or .nk file onto this file to open it.
cd /d "%~dp0"
set OFX_PLUGIN_PATH=%CD%\ofx-test-plugins;%OFX_PLUGIN_PATH%
start "" "%~dp0bin\strata_ui.exe" %*
