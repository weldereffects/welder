#include "../core/Parallel.h"
#include "ExrIO.h"

#include <fstream>
#include <filesystem>
#include <Iex.h>
#include <ImfIO.h>
#include <ImfChannelList.h>
#include <ImfCompression.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfStringAttribute.h>
#include <ImfInputPart.h>
#include <ImfMultiPartInputFile.h>
#include <ImfOutputFile.h>
#include <ImfPartType.h>
#include <ImfThreading.h>
#include <half.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace strata {

namespace {

void initThreads() {
    static std::once_flag once;
    std::call_once(once, [] { Imf::setGlobalThreadCount((int)std::max(1u, std::thread::hardware_concurrency())); });
}

const std::map<std::string, Imf::Compression>& compressionMap() {
    static const std::map<std::string, Imf::Compression> m = {
        {"none", Imf::NO_COMPRESSION},     {"rle", Imf::RLE_COMPRESSION},     {"zips", Imf::ZIPS_COMPRESSION},
        {"zip", Imf::ZIP_COMPRESSION},     {"piz", Imf::PIZ_COMPRESSION},     {"pxr24", Imf::PXR24_COMPRESSION},
        {"b44", Imf::B44_COMPRESSION},     {"b44a", Imf::B44A_COMPRESSION},   {"dwaa", Imf::DWAA_COMPRESSION},
        {"dwab", Imf::DWAB_COMPRESSION},   {"htj2k", Imf::HTJ2K32_COMPRESSION}, {"htj2k256", Imf::HTJ2K256_COMPRESSION}};
    return m;
}

std::string compressionName(Imf::Compression c) {
    for (auto& [k, v] : compressionMap())
        if (v == c) return k;
    return "unknown";
}

// Swap image rows top <-> bottom (EXR is y-down, Welder is y-up).
void flipRows(Image& img) {
    const int H = img.bounds.height();
    const size_t rowFloats = (size_t)img.bounds.width() * img.nch();
    parallelFor(H / 2, 16, [&](long i0, long i1) {
        std::vector<float> tmp(rowFloats);
        for (long i = i0; i < i1; ++i) {
            float* a = img.data.data() + (size_t)i * rowFloats;
            float* b = img.data.data() + (size_t)(H - 1 - i) * rowFloats;
            std::memcpy(tmp.data(), a, rowFloats * sizeof(float));
            std::memcpy(a, b, rowFloats * sizeof(float));
            std::memcpy(b, tmp.data(), rowFloats * sizeof(float));
        }
    });
}

}  // namespace

const std::vector<std::string>& exrCompressions() {
    static const std::vector<std::string> v = {"none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"};
    return v;
}

// The whole file, read in one go and decoded from memory. Network shares
// answer one large read far faster than the many small ones OpenEXR would
// make, and chunks then decode in parallel without a shared file position.
class MemoryStream : public Imf::IStream {
public:
    MemoryStream(const std::string& path, std::vector<char> data) : Imf::IStream(path.c_str()), d_(std::move(data)) {}
    bool isMemoryMapped() const override { return true; }
    bool read(char c[], int n) override {
        if (n < 0 || pos_ + (uint64_t)n > d_.size()) throw Iex::InputExc("Unexpected end of file.");
        std::memcpy(c, d_.data() + pos_, (size_t)n);
        pos_ += (uint64_t)n;
        return pos_ < d_.size();
    }
    char* readMemoryMapped(int n) override {
        if (n < 0 || pos_ + (uint64_t)n > d_.size()) throw Iex::InputExc("Unexpected end of file.");
        char* p = d_.data() + pos_;
        pos_ += (uint64_t)n;
        return p;
    }
    uint64_t tellg() override { return pos_; }
    void seekg(uint64_t pos) override { pos_ = pos; }
    void clear() override {}
    int64_t size() override { return (int64_t)d_.size(); }
    bool isStatelessRead() const override { return true; }
    int64_t read(void* buf, uint64_t sz, uint64_t offset) override {
        if (offset >= d_.size()) return 0;
        uint64_t n = std::min<uint64_t>(sz, d_.size() - offset);
        std::memcpy(buf, d_.data() + offset, (size_t)n);
        return (int64_t)n;
    }

private:
    std::vector<char> d_;
    uint64_t pos_ = 0;
};

std::vector<char> slurp(const std::string& path) {
    std::filesystem::path p(std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size()));
    std::ifstream f(p, std::ios::binary);
    if (!f) throw RenderError("cannot open " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n <= 0) throw RenderError("empty file: " + path);
    f.seekg(0);
    std::vector<char> d((size_t)n);
    if (!f.read(d.data(), n)) throw RenderError("cannot read " + path);
    return d;
}

ExrFile readExr(const std::string& path) {
    initThreads();
    try {
        MemoryStream stream(path, slurp(path));
        Imf::MultiPartInputFile in(stream, Imf::globalThreadCount());
        const int nparts = in.parts();
        if (nparts < 1) throw RenderError("EXR has no parts: " + path);

        const Imf::Header& h0 = in.header(0);
        const Imath::Box2i disp = h0.displayWindow();

        // Collect channels of every flat image part.
        struct Src {
            int part;
            std::string fileName;
            std::string name;
        };
        std::vector<Src> srcs;
        std::vector<int> imageParts;
        Imath::Box2i uni;  // union of data windows (EXR coordinates)
        bool haveBox = false;
        ExrFile out;
        for (int p = 0; p < nparts; ++p) {
            const Imf::Header& h = in.header(p);
            std::string pname = h.hasName() ? h.name() : "";
            out.parts.push_back(pname);
            if (h.hasType() && Imf::isDeepData(h.type())) continue;  // deep parts are skipped
            imageParts.push_back(p);
            const Imath::Box2i dw = h.dataWindow();
            if (!haveBox) {
                uni = dw;
                haveBox = true;
            } else {
                uni.extendBy(dw);
            }
            for (auto it = h.channels().begin(); it != h.channels().end(); ++it) {
                if (it.channel().xSampling != 1 || it.channel().ySampling != 1) continue;  // subsampled (YC) not supported
                std::string ch = it.name();
                std::string name = ch;
                bool mainPart = p == 0 || pname.empty() || pname == "rgba" || pname == "rgb";
                if (!mainPart && ch.find('.') == std::string::npos) name = pname + "." + ch;
                srcs.push_back({p, ch, name});
            }
        }
        if (imageParts.empty()) throw RenderError("EXR contains only deep data (not supported yet): " + path);
        out.compression = compressionName(in.header(imageParts[0]).compression());

        // Channel order: R, G, B, A, then everything else as stored.
        ChannelSet cs;
        std::vector<int> order;
        for (const char* m : {"R", "G", "B", "A"})
            for (size_t i = 0; i < srcs.size(); ++i)
                if (srcs[i].name == m && channelIndex(cs, m) < 0) {
                    cs.push_back(m);
                    order.push_back((int)i);
                }
        for (size_t i = 0; i < srcs.size(); ++i)
            if (channelIndex(cs, srcs[i].name) < 0) {
                cs.push_back(srcs[i].name);
                order.push_back((int)i);
            }
        bool lumaOnly = channelIndex(cs, "R") < 0 && channelIndex(cs, "Y") >= 0;

        out.format.width = disp.max.x - disp.min.x + 1;
        out.format.height = disp.max.y - disp.min.y + 1;
        out.format.pixelAspect = h0.pixelAspectRatio() > 0 ? h0.pixelAspectRatio() : 1.0;

        // Welder coordinates: x - disp.min.x, y = disp.max.y - y_exr.
        Box bounds(uni.min.x - disp.min.x, disp.max.y - uni.max.y, uni.max.x - disp.min.x + 1, disp.max.y - uni.min.y + 1);
        ChannelSet finalCs = cs;
        if (lumaOnly) finalCs.insert(finalCs.begin(), {"R", "G", "B"});
        auto img = std::make_shared<Image>(bounds, finalCs);
        const int n = img->nch();
        const int W = bounds.width();
        const int off = lumaOnly ? 3 : 0;

        // Read each part straight into the interleaved buffer, rows top-down, then flip once.
        for (int p : imageParts) {
            const Imf::Header& h = in.header(p);
            const Imath::Box2i dw = h.dataWindow();
            Imf::FrameBuffer fb;
            for (size_t k = 0; k < order.size(); ++k) {
                const Src& s = srcs[order[k]];
                if (s.part != p) continue;
                char* first = reinterpret_cast<char*>(img->data.data() +
                                                      ((size_t)(dw.min.y - uni.min.y) * W + (dw.min.x - uni.min.x)) * n +
                                                      (k + off));
                fb.insert(s.fileName, Imf::Slice::Make(Imf::FLOAT, first, dw, sizeof(float) * n, sizeof(float) * n * W));
            }
            Imf::InputPart part(in, p);
            part.setFrameBuffer(fb);
            part.readPixels(dw.min.y, dw.max.y);
        }
        flipRows(*img);
        if (lumaOnly) {
            int y = img->ch("Y");
            for (size_t i = 0; i < (size_t)bounds.area(); ++i) {
                float* px = img->data.data() + i * n;
                px[0] = px[1] = px[2] = px[y];
            }
        }
        out.image = img;
        for (int p = 0; p < nparts; ++p)
            for (auto it = in.header(p).begin(); it != in.header(p).end(); ++it)
                if (auto* sa = dynamic_cast<const Imf::StringAttribute*>(&it.attribute())) out.metadata[it.name()] = sa->value();
        return out;
    } catch (const RenderError&) {
        throw;
    } catch (const std::exception& e) {
        throw RenderError("cannot read EXR " + path + ": " + e.what());
    }
}

void writeExr(const std::string& path, const Image& src, const Format& fmt, const ExrWriteOptions& o) {
    initThreads();
    auto cit = compressionMap().find(o.compression.empty() ? "zip" : o.compression);
    if (cit == compressionMap().end()) throw RenderError("unknown EXR compression '" + o.compression + "'");
    if (src.channels.empty()) throw RenderError("nothing to write (no channels): " + path);

    // Data window in Welder coordinates, then in EXR coordinates (y-down from the top of the format).
    Box dwS = o.dataWindowFromBbox && !src.bounds.empty() ? src.bounds : Box(0, 0, fmt.width, fmt.height);
    Imath::Box2i disp(Imath::V2i(0, 0), Imath::V2i(fmt.width - 1, fmt.height - 1));
    Imath::Box2i dw(Imath::V2i(dwS.x1, fmt.height - dwS.y2), Imath::V2i(dwS.x2 - 1, fmt.height - 1 - dwS.y1));

    Image buf(dwS, src.channels);
    buf.copyFrom(src);
    flipRows(buf);  // now row 0 is the top (EXR order)

    try {
        Imf::Header h(disp, dw, (float)fmt.pixelAspect);
        h.compression() = cit->second;
        if (cit->second == Imf::DWAA_COMPRESSION || cit->second == Imf::DWAB_COMPRESSION) h.dwaCompressionLevel() = o.dwaLevel;
        const bool half = o.type == ExrPixelType::Half;
        Imf::PixelType pt = half ? Imf::HALF : Imf::FLOAT;
        Imf::FrameBuffer fb;
        const int n = buf.nch(), W = dwS.width();
        // The library does not convert pixel types on write, so halves are made here.
        std::vector<Imath::half> hbuf;
        if (half) {
            hbuf.resize(buf.data.size());
            for (size_t i = 0; i < buf.data.size(); ++i) hbuf[i] = Imath::half(buf.data[i]);
        }
        for (int c = 0; c < n; ++c) {
            h.channels().insert(buf.channels[c], Imf::Channel(pt));
            if (half)
                fb.insert(buf.channels[c], Imf::Slice::Make(Imf::HALF, hbuf.data() + c, dw, sizeof(Imath::half) * n,
                                                            sizeof(Imath::half) * n * W));
            else
                fb.insert(buf.channels[c], Imf::Slice::Make(Imf::FLOAT, buf.data.data() + c, dw, sizeof(float) * n,
                                                            sizeof(float) * n * W));
        }
        Imf::OutputFile out(path.c_str(), h, Imf::globalThreadCount());
        out.setFrameBuffer(fb);
        out.writePixels(dw.max.y - dw.min.y + 1);
    } catch (const std::exception& e) {
        throw RenderError("cannot write EXR " + path + ": " + e.what());
    }
}

}  // namespace strata
