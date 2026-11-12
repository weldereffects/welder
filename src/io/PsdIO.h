// Photoshop documents (.psd, and .psb for large documents), read only (issue #12).
//
// The composite comes in as R, G, B, A, and every layer as channels of its own, named after it
// (a layer "Sky glow" is Sky_glow.R / .G / .B / .A), placed where it sits on the canvas. Hidden
// layers are read too. Each layer's visibility, opacity and blend mode are in the metadata
// ("psd/<layer>": "visible=1 opacity=0.85 blend=norm"), and "psd/layers" lists them bottom to top.
//
// RGB and grayscale documents at 8, 16 or 32 bits; raw, RLE and zip (with or without prediction)
// compressed data. 8 and 16-bit pixels go through `decode` (or sRGB) to linear like other integer
// formats; 32-bit documents are linear already. Group folders, adjustment layers, layer masks and
// effects aren't applied: the layers' own pixels are what you get.
#pragma once

#include <string>

#include "ImageIO.h"

namespace strata {

bool isPsdPath(const std::string& path);
// Throws RenderError on failure (not a PSD, a colour mode other than RGB / grayscale, truncated).
ExrFile readPsd(const std::string& path, const ReadOptions& o = {});

}  // namespace strata
