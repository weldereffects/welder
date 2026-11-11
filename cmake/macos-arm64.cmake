# Apple silicon (arm64) builds on an Intel Mac: CI builds the arm64 Welder.app on GitHub's Intel
# machines, because its Apple silicon machines are often all taken. Picked up through the
# CMAKE_TOOLCHAIN_FILE environment variable (tools/package_macos.sh with WELDER_MAC_ARCH=arm64), so
# the dependencies' builds (OpenColorIO and its libraries, ncnn) cross-compile the same way.
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR arm64)
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "")
