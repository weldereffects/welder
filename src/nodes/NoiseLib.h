// Procedural noise for the Noise node: gradient (Perlin) and value noise on a
// seeded integer hash, Worley cells, and fractal / warped combinations of them.
// Values come out roughly in 0..1 (each type is normalised so its usual range
// fills it). Coordinates are in "feature" units: one unit = the noise size.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::noise {

enum class Type {
    Perlin,      // gradient noise fBm: soft clouds
    Value,       // value noise fBm: softer, slightly blobby
    Blocky,      // value noise without interpolation: stepped pixels at every octave
    Turbulence,  // sum of |noise|: billowy, creased
    Ridged,      // ridged multifractal: sharp ridges, mountain lines
    Hybrid,      // hybrid multifractal: smooth valleys, rough peaks
    Blistered,   // turbulence raised to a power: sparse blisters
    Gas,         // fBm warped by fBm warped by fBm: smoky swirls
    Wavy,        // sine bands bent by turbulence: marble, wood, waves
    Electro,     // thin bright veins: lightning, caustic lines
    Plasma,      // interfering sines warped by noise: classic plasma
    CellF1,      // Worley F1: distance to the nearest point (soft cells)
    CellF2,      // Worley F2: distance to the second nearest
    CellEdges,   // F2 - F1: cell borders (cracks, scales)
    Cells,       // flat random value per cell (voronoi mosaic)
    Crackle,     // thin cell borders (dried mud, crackle)
    Dents,       // rounded dents per cell (hammered metal)
    Dots,        // scattered round spots of random size (sparse convolution)
    Random,      // white noise (blocks of `size`)
};

struct TypeInfo {
    Type type;
    const char* name;  // the knob value
};
const std::vector<TypeInfo>& types();
Type typeFromName(const std::string& n);  // unknown -> Perlin

enum class Metric { Euclidean, Manhattan, Chebyshev };

struct Settings {
    Type type = Type::Perlin;
    uint32_t seed = 0;
    float octaves = 4;        // fractional: the last octave fades in
    float lacunarity = 2;     // frequency step per octave
    float gain = 0.5f;        // amplitude step per octave
    float distortion = 0;     // domain warp amount, in feature units
    float distortionScale = 1;
    float jitter = 1;         // cells: how far points stray from the grid (0..1)
    Metric metric = Metric::Euclidean;
    float sharpness = 8;      // Electro: line thinness; Blistered: exponent / 2
    float waves = 1;          // Wavy / Plasma: bands per feature
    float density = 0.5f;     // Dots: chance of a dot per cell
    // Seamless loop: > 0 evolves w around a circle of this circumference
    // (gradient types: true 4D noise; others: crossfade).
    float loopLength = 0;
};

// The noise at (x, y) with evolution z (and the loop phase, 0..1, when looping).
float eval(const Settings& s, float x, float y, float z, float loopPhase = 0);

// How many octaves are worth computing for features `sizePx` pixels across
// (finer ones would alias, Nyquist): min(octaves, ...).
float bandLimit(const Settings& s, float sizePx);

// Raw bases (exposed for tests): roughly -1..1.
float perlin3(float x, float y, float z, uint32_t seed);
float perlin4(float x, float y, float z, float w, uint32_t seed);
float value3(float x, float y, float z, uint32_t seed);

}  // namespace strata::noise
