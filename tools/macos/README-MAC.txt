Strata for macOS
================

Unzip, then drag Strata.app to Applications (or run it from anywhere).
There are two downloads: -arm64 for Apple silicon (M1 and later) and
-x86_64 for Intel Macs.

FIRST LAUNCH
  This build isn't notarized by Apple yet, so the first time:
  right-click (or Control-click) Strata.app > Open > Open.
  After that it opens normally.

WHAT IT NEEDS
  - macOS 11 (Big Sur) or later.
  - Optional: Python 3 (python.org or Homebrew) for the Nuke importer
    (File > Import .nk, pasting nodes copied in Nuke) and, after
    Strata.app/Contents/Resources/ml/setup-ml.sh, the AI nodes.
  - Optional: GPU processing uses Vulkan through MoltenVK
    (brew install molten-vk); without it everything runs on the CPU.

THE COMMAND LINE RENDERER
  Strata.app/Contents/MacOS/strata comp.strata    renders the comp's Writes

OFX PLUGINS
  Strata looks in /Library/OFX/Plugins and in OFX_PLUGIN_PATH.

Everything else (hotkeys, nodes, Nuke import) is as on Windows: see
README.md and docs/.
