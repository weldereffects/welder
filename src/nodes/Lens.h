// Lens distortion: a radial (Brown) model, k1 r^2 + k2 r^4 + k3 r^6, about a
// centre, with an optional anamorphic squeeze. Radii are normalised by half the
// image diagonal, so the coefficients don't depend on the resolution (or the
// render scale). Positive k1 = pincushion, negative = barrel.
//
// distort():   undistorted (ideal pinhole) pixel -> where it lands in the plate
// undistort(): plate pixel -> ideal pinhole pixel (the inverse, by Newton)
//
// Pixel coordinates are Strata's (origin bottom-left, y up), in a W x H image.
#pragma once

#include <algorithm>
#include <cmath>

namespace strata::lens {

struct Lens {
    double k1 = 0, k2 = 0, k3 = 0;
    double cx = 0, cy = 0;   // centre offset, in units of the normalisation radius
    double squeeze = 1;      // anamorphic: x is divided by it before the radial term

    bool none() const { return k1 == 0 && k2 == 0 && k3 == 0; }

    static double radius(double W, double H) { return 0.5 * std::sqrt(W * W + H * H); }

    // radial factor at normalised radius^2
    double factor(double r2) const { return 1 + r2 * (k1 + r2 * (k2 + r2 * k3)); }

    void distort(double x, double y, double W, double H, double& ox, double& oy) const {
        const double R = radius(W, H);
        double nx = (x - W * 0.5) / R - cx, ny = (y - H * 0.5) / R - cy;
        nx /= squeeze;
        double f = factor(nx * nx + ny * ny);
        ox = (nx * f * squeeze + cx) * R + W * 0.5;
        oy = (ny * f + cy) * R + H * 0.5;
    }

    void undistort(double x, double y, double W, double H, double& ox, double& oy) const {
        const double R = radius(W, H);
        double nx = (x - W * 0.5) / R - cx, ny = (y - H * 0.5) / R - cy;
        nx /= squeeze;
        double rd = std::sqrt(nx * nx + ny * ny);
        double ru = rd;
        if (rd > 1e-12) {
            // solve ru * factor(ru^2) = rd
            for (int i = 0; i < 20; ++i) {
                double r2 = ru * ru;
                double g = ru * factor(r2) - rd;
                double dg = 1 + r2 * (3 * k1 + r2 * (5 * k2 + r2 * 7 * k3));
                if (std::fabs(dg) < 1e-9) break;
                double step = g / dg;
                ru -= step;
                if (ru < 0) ru = 0.5 * (ru + step);
                if (std::fabs(step) < 1e-12) break;
            }
        }
        double s = rd > 1e-12 ? ru / rd : 1;
        ox = (nx * s * squeeze + cx) * R + W * 0.5;
        oy = (ny * s + cy) * R + H * 0.5;
    }
};

}  // namespace strata::lens
