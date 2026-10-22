#include "FrameList.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

namespace strata {

namespace {
// A number at s[i] (optionally signed); advances i.
bool number(const std::string& s, size_t& i, long& v) {
    size_t j = i;
    if (j < s.size() && (s[j] == '-' || s[j] == '+')) ++j;
    size_t d = j;
    while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
    if (j == d) return false;
    v = std::strtol(s.substr(i, j - i).c_str(), nullptr, 10);
    i = j;
    return true;
}
}  // namespace

bool parseFrameList(const std::string& text, std::vector<double>& out, std::string* err) {
    out.clear();
    // items are separated by commas or spaces
    std::string s = text;
    for (auto& c : s)
        if (c == ',' || c == '\t' || c == ';') c = ' ';
    size_t i = 0;
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        out.clear();
        return false;
    };
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        if (i >= s.size()) break;
        long a = 0, b = 0, step = 1;
        if (!number(s, i, a)) return fail("expected a frame number at \"" + s.substr(i) + "\"");
        b = a;
        if (i < s.size() && s[i] == '-') {
            ++i;
            if (!number(s, i, b)) return fail("expected the last frame after '-'");
        }
        if (i < s.size() && (s[i] == 'x' || s[i] == 'X' || s[i] == 'y' || s[i] == 'Y')) {
            ++i;
            if (!number(s, i, step) || step <= 0) return fail("expected a step after 'x'");
        }
        if (i < s.size() && s[i] != ' ') return fail("unexpected '" + std::string(1, s[i]) + "'");
        if (b < a) std::swap(a, b);
        if ((b - a) / step > 1000000) return fail("too many frames");
        for (long f = a; f <= b; f += step) out.push_back((double)f);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.empty()) return fail("no frames");
    return true;
}

std::string frameRangeText(double first, double last) {
    long a = std::lround(first), b = std::lround(last);
    return a == b ? std::to_string(a) : std::to_string(a) + "-" + std::to_string(b);
}

}  // namespace strata
