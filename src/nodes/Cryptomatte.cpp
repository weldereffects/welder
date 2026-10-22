#include "Cryptomatte.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "../core/Engine.h"
#include "nlohmann/json.hpp"

namespace strata::crypto {

uint32_t murmur3(const std::string& s, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)s.data();
    const int len = (int)s.size(), nblocks = len / 4;
    uint32_t h = seed;
    const uint32_t c1 = 0xcc9e2d51, c2 = 0x1b873593;
    auto rotl = [](uint32_t x, int r) { return (x << r) | (x >> (32 - r)); };
    for (int i = 0; i < nblocks; ++i) {
        uint32_t k;
        std::memcpy(&k, data + i * 4, 4);  // little-endian, as the reference
        k *= c1;
        k = rotl(k, 15);
        k *= c2;
        h ^= k;
        h = rotl(h, 13);
        h = h * 5 + 0xe6546b64;
    }
    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16; [[fallthrough]];
        case 2: k1 ^= tail[1] << 8; [[fallthrough]];
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = rotl(k1, 15);
            k1 *= c2;
            h ^= k1;
    }
    h ^= (uint32_t)len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

float idOf(const std::string& name) {
    uint32_t h = murmur3(name);
    uint32_t exp = (h >> 23) & 255;
    if (exp == 0 || exp == 255) h ^= 1u << 23;  // no denormals, infinities or NaNs
    float f;
    std::memcpy(&f, &h, 4);
    return f;
}

uint32_t bitsOf(float id) {
    uint32_t b;
    std::memcpy(&b, &id, 4);
    return b;
}

std::string hexOf(float id) {
    char b[16];
    std::snprintf(b, sizeof b, "%08x", bitsOf(id));
    return b;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r"), b = s.find_last_not_of(" \t\n\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::vector<std::string> splitList(const std::string& list) {
    // comma separated; names may be quoted ("a, b") to hold commas
    std::vector<std::string> out;
    std::string cur;
    bool q = false;
    for (char c : list) {
        if (c == '"') { q = !q; continue; }
        if (c == ',' && !q) {
            if (!trim(cur).empty()) out.push_back(trim(cur));
            cur.clear();
        } else
            cur += c;
    }
    if (!trim(cur).empty()) out.push_back(trim(cur));
    return out;
}

std::string joinList(const std::vector<std::string>& items) {
    std::string s;
    for (auto& i : items) {
        if (!s.empty()) s += ", ";
        s += i.find(',') != std::string::npos ? "\"" + i + "\"" : i;
    }
    return s;
}

bool entryId(const std::string& e, float& id) {
    if (e.size() == 10 && e.front() == '<' && e.back() == '>') {
        uint32_t b = (uint32_t)std::strtoul(e.substr(1, 8).c_str(), nullptr, 16);
        std::memcpy(&id, &b, 4);
        return true;
    }
    if (e.empty()) return false;
    id = idOf(e);
    return true;
}

namespace {
std::mutex gM;
std::map<uint32_t, std::pair<std::string, std::string>> gNames;  // id bits -> (name, layer)
}  // namespace

void registerMetadata(const std::map<std::string, std::string>& md) {
    // cryptomatte/<key>/name = layer name; cryptomatte/<key>/manifest = {"name": "hex", ...}
    std::map<std::string, std::string> layerOf;
    for (auto& [k, v] : md)
        if (k.rfind("cryptomatte/", 0) == 0 && k.size() > 5 && k.compare(k.size() - 5, 5, "/name") == 0)
            layerOf[k.substr(12, k.size() - 17)] = v;
    for (auto& [k, v] : md) {
        if (k.rfind("cryptomatte/", 0) != 0 || k.size() < 9 || k.compare(k.size() - 9, 9, "/manifest") != 0) continue;
        std::string key = k.substr(12, k.size() - 21);
        std::string layer = layerOf.count(key) ? layerOf[key] : "";
        try {
            auto j = nlohmann::json::parse(v);
            std::lock_guard<std::mutex> l(gM);
            for (auto& [name, hex] : j.items()) {
                if (!hex.is_string()) continue;
                uint32_t b = (uint32_t)std::strtoul(hex.get<std::string>().c_str(), nullptr, 16);
                gNames[b] = {name, layer};
            }
        } catch (...) {
        }
    }
}

std::string nameFor(float id) {
    std::lock_guard<std::mutex> l(gM);
    auto it = gNames.find(bitsOf(id));
    return it != gNames.end() ? it->second.first : "<" + hexOf(id) + ">";
}

std::vector<std::pair<std::string, float>> knownNames(const std::string& layer) {
    std::vector<std::pair<std::string, float>> out;
    std::lock_guard<std::mutex> l(gM);
    for (auto& [b, nl] : gNames) {
        if (!layer.empty() && !nl.second.empty()) {
            std::string a = nl.second, c = layer;
            for (auto& ch : a) ch = (char)std::tolower((unsigned char)ch);
            for (auto& ch : c) ch = (char)std::tolower((unsigned char)ch);
            a.erase(std::remove(a.begin(), a.end(), '_'), a.end());
            c.erase(std::remove(c.begin(), c.end(), '_'), c.end());
            if (a.find(c) == std::string::npos && c.find(a) == std::string::npos) continue;
        }
        float f;
        std::memcpy(&f, &b, 4);
        out.emplace_back(nl.first, f);
    }
    std::sort(out.begin(), out.end());
    return out;
}

Layer findLayer(const ChannelSet& cs, const std::string& layer) {
    Layer L;
    for (int i = 0; i < 32; ++i) {
        char base[96];
        std::snprintf(base, sizeof base, "%s%02d.", layer.c_str(), i);
        std::string b = base;
        int r = channelIndex(cs, b + "R"), g = channelIndex(cs, b + "G"), bl = channelIndex(cs, b + "B"), a = channelIndex(cs, b + "A");
        if (r < 0 || g < 0) break;
        L.id.push_back(r);
        L.cov.push_back(g);
        if (bl >= 0 && a >= 0) {
            L.id.push_back(bl);
            L.cov.push_back(a);
        }
    }
    return L;
}

bool topId(const float* px, const Layer& L, float& id) {
    float best = 0;
    bool any = false;
    for (size_t r = 0; r < L.id.size(); ++r)
        if (px[L.cov[r]] > best) {
            best = px[L.cov[r]];
            id = px[L.id[r]];
            any = true;
        }
    return any;
}

// ------------------------------------------------------------------ the keyer
namespace {

class CryptomatteNode : public Node {
public:
    int maxInputs() const override { return 1; }
    ChannelSet channels(Engine& e, double f) override {
        ChannelSet c = input(0) ? e.channels(input(0), f) : rgbaChannels();
        c = unionChannels(c, rgbaChannels());
        if (params.str("output", "alpha") == "matte channel") c = unionChannels(c, {"matte.A"});
        return c;
    }
    void render(RenderArgs& a, Image& out) override {
        std::fill(out.data.begin(), out.data.end(), 0.0f);
        if (!input(0)) return;
        ImagePtr src = a.fetch(input(0), a.region);
        out.copyFrom(*src);
        std::string layer = params.str("layer", "cryptoobject");
        Layer L = findLayer(src->channels, layer);
        if (L.id.empty())
            throw RenderError(name + ": the input has no " + layer + "00 channels (enable cryptomatte on the render, and pick its layer)");
        std::vector<uint32_t> ids;
        for (auto& e : splitList(params.str("matte_list", ""))) {
            float id;
            if (entryId(e, id)) ids.push_back(bitsOf(id));
        }
        std::sort(ids.begin(), ids.end());
        const bool inv = params.flag("invert", false), preview = params.flag("preview", false);
        const std::string mode = params.str("output", "alpha");
        const int A = out.ch("A"), M = out.ch("matte.A"), R = out.ch("R"), G = out.ch("G"), B = out.ch("B");
        const int sn = src->nch(), on = out.nch();
        for (int y = a.region.y1; y < a.region.y2; ++y)
            for (int x = a.region.x1; x < a.region.x2; ++x) {
                const float* s = src->px(x, y);
                float* o = out.px(x, y);
                float m = 0;
                float col[3] = {0, 0, 0};
                for (size_t r = 0; r < L.id.size(); ++r) {
                    float cov = s[L.cov[r]];
                    if (cov <= 0) continue;
                    uint32_t b = bitsOf(s[L.id[r]]);
                    bool hit = std::binary_search(ids.begin(), ids.end(), b);
                    if (hit) m += cov;
                    if (preview) {  // a colour per id, from its bits
                        float k = hit ? 1.0f : 0.35f;
                        col[0] += cov * k * ((b & 0xff) / 255.0f);
                        col[1] += cov * k * (((b >> 8) & 0xff) / 255.0f);
                        col[2] += cov * k * (((b >> 16) & 0xff) / 255.0f);
                    }
                }
                m = std::clamp(m, 0.0f, 1.0f);
                if (inv) m = 1 - m;
                if (mode == "matte channel") o[M] = m;
                else if (mode == "premultiplied rgba") {  // the matte as alpha, colour cut out by it
                    for (int c : {R, G, B})
                        if (c >= 0) o[c] *= m;
                    if (A >= 0) o[A] = m;
                } else if (A >= 0)
                    o[A] = m;
                if (preview && R >= 0 && G >= 0 && B >= 0) {
                    float base = 0.2126f * o[R] + 0.7152f * o[G] + 0.0722f * o[B];
                    o[R] = 0.5f * base + col[0] * 0.8f + (m > 0 ? 0.3f * m : 0);
                    o[G] = 0.5f * base + col[1] * 0.8f + (m > 0 ? 0.3f * m : 0);
                    o[B] = 0.5f * base + col[2] * 0.8f;
                }
                (void)sn;
                (void)on;
            }
    }
};

}  // namespace

}  // namespace strata::crypto

namespace strata {
void registerCryptomatteNode() {
    NodeRegistry::instance().add("Cryptomatte", [] { return std::make_unique<crypto::CryptomatteNode>(); });
}
}  // namespace strata
