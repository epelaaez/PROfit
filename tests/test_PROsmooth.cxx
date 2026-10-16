// Standalone test of inc/PROsmooth.h (Eigen only, no ROOT). Build and run:
//   g++ -std=c++17 -O2 -I inc -I /usr/include/eigen3 tests/test_PROsmooth.cxx -o test_PROsmooth && ./test_PROsmooth
#include "PROsmooth.h"

#include <cstdio>
#include <limits>
#include <random>

using PROfit::PROsmooth::SmoothBlock;

static int failures = 0;
static void check(bool ok, const char *what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const float tol = 1e-5f;

    {   // A constant ratio is unchanged.
        Eigen::VectorXf y = Eigen::VectorXf::Constant(12, 1.07f), w = Eigen::VectorXf::Constant(12, 50.f);
        SmoothBlock(y, w, w, {12}, 1.5f);
        check((y.array() - 1.07f).abs().maxCoeff() < tol, "constant ratio unchanged");
    }
    {   // A linear trend is reproduced exactly, edges included, even with uneven weights.
        const int n = 15;
        Eigen::VectorXf y(n), truth(n), w(n);
        for (int i = 0; i < n; ++i) { truth(i) = 0.9f + 0.02f * i; w(i) = 10.f + 7.f * (i % 4); }
        y = truth;
        SmoothBlock(y, w, w, {(size_t)n}, 2.0f, false);
        check((y - truth).cwiseAbs().maxCoeff() < tol, "linear trend reproduced, including edges");
    }
    {   // sigma <= 0 is a no-op.
        Eigen::VectorXf y = Eigen::VectorXf::LinSpaced(8, 0.5f, 1.5f), y0 = y, w = Eigen::VectorXf::Ones(8);
        y(3) = 2.0f; y0(3) = 2.0f;
        SmoothBlock(y, w, w, {8}, 0.0f);
        check(y == y0, "sigma = 0 leaves the ratio unchanged");
    }
    {   // Empty-CV bins keep their value and do not leak into neighbours.
        Eigen::VectorXf y = Eigen::VectorXf::Constant(9, 1.0f), w = Eigen::VectorXf::Constant(9, 20.f);
        y(4) = 5.0f; w(4) = 0.0f;
        SmoothBlock(y, w, w, {9}, 1.0f);
        bool ok = std::abs(y(4) - 5.0f) < tol;
        for (int i = 0; i < 9; ++i) if (i != 4) ok = ok && std::abs(y(i) - 1.0f) < tol;
        check(ok, "zero-weight bin untouched and isolated");
    }
    {   // Rate preservation: sum(cv * (y - 1)) is the same before and after.
        std::mt19937 rng(1);
        std::normal_distribution<float> g(0.f, 0.05f);
        const int n = 20;
        Eigen::VectorXf y(n), cv(n);
        for (int i = 0; i < n; ++i) { cv(i) = 100.f * std::exp(-0.1f * i) + 5.f; y(i) = 1.0f + 0.1f * std::sin(0.4f * i) + g(rng); }
        const double before = (cv.array() * (y.array() - 1.f)).sum();
        SmoothBlock(y, cv, cv, {(size_t)n}, 1.5f, true);
        const double after = (cv.array() * (y.array() - 1.f)).sum();
        check(std::abs(before - after) < 1e-3 * std::abs(before) + 1e-4, "CV-weighted rate change preserved");
    }
    {   // 2D (row-major, last dim fastest): a plane is reproduced exactly.
        const size_t nx = 6, ny = 4;
        Eigen::VectorXf y(nx * ny), truth(nx * ny), w = Eigen::VectorXf::Constant(nx * ny, 30.f);
        for (size_t ix = 0; ix < nx; ++ix)
            for (size_t iy = 0; iy < ny; ++iy) truth(ix * ny + iy) = 1.f + 0.03f * ix - 0.05f * iy;
        y = truth;
        SmoothBlock(y, w, w, {nx, ny}, 1.2f, false);
        check((y - truth).cwiseAbs().maxCoeff() < tol, "2D plane reproduced (axis order correct)");
    }
    {   // Toy MC: Poisson-fluctuated variation over a smooth truth. Smoothing should bring
        // the ratio closer to the truth on average.
        std::mt19937 rng(7);
        const int n = 30, ntoys = 200;
        double rms_raw = 0, rms_sm = 0;
        for (int t = 0; t < ntoys; ++t) {
            Eigen::VectorXf cv(n), y(n), truth(n);
            for (int i = 0; i < n; ++i) {
                const float mu = 400.f * std::exp(-std::pow((i - 12.f) / 8.f, 2)) + 20.f;
                truth(i) = 1.f + 0.08f * std::tanh((i - 15.f) / 6.f);
                std::poisson_distribution<int> pc(mu), pv(mu * truth(i));
                cv(i) = (float)pc(rng);
                const float v = (float)pv(rng);
                y(i) = cv(i) > 0 ? v / cv(i) : 1.f;
            }
            rms_raw += (y - truth).squaredNorm();
            SmoothBlock(y, cv, cv, {(size_t)n}, 1.5f, true);
            rms_sm += (y - truth).squaredNorm();
        }
        rms_raw = std::sqrt(rms_raw / (n * ntoys));
        rms_sm = std::sqrt(rms_sm / (n * ntoys));
        std::printf("       toy RMS(ratio - truth): raw %.4f -> smoothed %.4f\n", rms_raw, rms_sm);
        check(rms_sm < 0.7 * rms_raw, "smoothing reduces the ratio error on Poisson toys");
    }

    {   // FirstInvalidSmoothed: a ratio falling to zero at the edge makes the local-linear fit
        // extrapolate below zero in the last bin, which FillSpline must refuse.
        using PROfit::PROsmooth::FirstInvalidSmoothed;
        Eigen::VectorXf y(10), cv = Eigen::VectorXf::Constant(10, 10.f);
        y << 0.40f, 0.35f, 0.30f, 0.25f, 0.20f, 0.15f, 0.10f, 0.05f, 0.0f, 0.0f;
        const Eigen::VectorXf raw = y;
        SmoothBlock(y, cv, cv, {10}, 1.0f, true);
        check(FirstInvalidSmoothed(raw, y) == 9 && y(9) < 0.0f, "ratio running to zero at the edge is flagged");

        Eigen::VectorXf r(3), sm(3);
        r << 1.0f, -0.2f, 0.9f;  sm << 1.0f, -0.1f, 0.95f;   // pre-existing negative: not smoothing's doing
        check(FirstInvalidSmoothed(r, sm) == -1, "valid smoothing and pre-existing negatives pass");
        sm(2) = std::numeric_limits<float>::quiet_NaN();
        check(FirstInvalidSmoothed(r, sm) == 2, "NaN is flagged");
    }

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
