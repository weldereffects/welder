# Releasing Welder

## Version numbers

Each update that goes out to testers raises the last number: 0.1.0, 0.1.1, 0.1.2 ... The version is
`project(welder VERSION x.y.z)` in `CMakeLists.txt`; the app (Help › About), the command-line tool and
the package names all take it from there. (`WELDER_PRERELEASE` adds a label such as `beta.1`; it is
empty now.)

For each version, write `docs/release-notes/<version>.md`: what changed since the last one.

## Builds

- Windows: `tools/package_windows.sh` (cross-compiled with MinGW-w64).
- Linux: the `build` workflow on GitHub (AlmaLinux 8, glibc 2.28: runs on Rocky / Alma / RHEL 8 and 9
  and every newer distribution). `tools/package_linux.sh` on a newer system makes a package that needs
  that system's glibc.
- macOS: the `build` workflow (Apple silicon and Intel), when run by hand or for a tag.

Pushing a `v<version>` tag (or publishing a GitHub release with that tag) runs every build and
attaches the packages to the release.
