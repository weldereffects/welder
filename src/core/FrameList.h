// Frame lists as the .nk format writes them: "1040-1105", "1-100x2" (every 2nd),
// "1,5,10", "1-10 20-30", single frames, negative frames ("-5--1").
#pragma once

#include <string>
#include <vector>

namespace strata {

// Sorted, without repeats. False (and err) on a malformed list.
bool parseFrameList(const std::string& text, std::vector<double>& out, std::string* err = nullptr);
// "1040-1105" (or "7" for one frame).
std::string frameRangeText(double first, double last);

}  // namespace strata
