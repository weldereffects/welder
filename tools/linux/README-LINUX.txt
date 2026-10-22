Strata for Linux (x86_64)
=========================

No install needed. Unpack anywhere. The download comes in up to three
tarballs that all unpack into the same folder: unpack them in the same place.
  strata-<version>-linux-x86_64.tar.gz              the app (required)
  strata-<version>-linux-x86_64-rife-models.tar.gz  the RIFE node's networks
  strata-<version>-linux-x86_64-tests.tar.gz        self-tests (optional)

    tar xzf strata-<version>-linux-x86_64.tar.gz
    cd strata-<version>-linux-x86_64
    ./Strata

  ./install-desktop-entry.sh   adds Strata to your applications menu.

WHAT IT NEEDS
  - A 64-bit distribution with glibc as new as the one it was built on
    (this build: glibc 2.38 or newer - Ubuntu 24.04+, Fedora 39+, Debian 13+,
    Arch, Mint 22+). Older studio systems (Rocky / Alma 8 and 9) need a
    build made on them.
  - OpenGL 3.3 (any current NVIDIA, AMD or Intel driver) and X11 or XWayland.
  - Optional: Vulkan (libvulkan.so.1) for GPU processing; without it
    everything runs on the CPU.
  - Optional: Python 3 for the Nuke importer (File > Import .nk, pasting
    nodes copied in Nuke) and, after bin/ml/setup-ml.sh, the AI nodes.

THE COMMAND LINE RENDERER
  bin/strata comp.strata            renders the comp's Write nodes
  bin/strata --help                 everything else

Everything else (hotkeys, nodes, Nuke import) is as on Windows: see
README.md and docs/.
