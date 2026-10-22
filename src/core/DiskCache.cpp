#include "DiskCache.h"

#include "CacheDir.h"

#include <half.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "Node.h"

namespace fs = std::filesystem;

namespace strata {

namespace {
const char kMagic[4] = {'S', 'D', 'C', '1'};
const char* kExt = ".sdc";
constexpr size_t kMaxQueued = 4;  // frames waiting to be written; more are skipped

bool parseKey(const std::string& stem, uint64_t& k) {
    if (stem.size() != 16) return false;
    char* end = nullptr;
    k = std::strtoull(stem.c_str(), &end, 16);
    return end && *end == 0;
}
}  // namespace

std::string DiskCache::defaultDir() { return (std::filesystem::path(cacheRoot()) / "frames").string(); }

uint64_t DiskCache::key(uint64_t nodeHash, const Box& r) {
    uint64_t h = hashString("strata-disk-cache-1 " STRATA_VERSION, nodeHash);
    int b[4] = {r.x1, r.y1, r.x2, r.y2};
    return hashBytes(b, sizeof b, h);
}

DiskCache::DiskCache(std::string dir, size_t budgetBytes) : dir_(std::move(dir)), budget_(budgetBytes) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    // what's there from earlier sessions, oldest first
    std::vector<std::pair<fs::file_time_type, std::pair<uint64_t, size_t>>> found;
    for (auto it = fs::directory_iterator(dir_, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        const fs::path& p = it->path();
        if (p.extension() == ".tmp") {  // an interrupted write
            fs::remove(p, ec);
            continue;
        }
        uint64_t k;
        if (p.extension() != kExt || !parseKey(p.stem().string(), k)) continue;
        std::error_code e2;
        size_t sz = (size_t)fs::file_size(p, e2);
        auto t = fs::last_write_time(p, e2);
        if (!e2) found.push_back({t, {k, sz}});
    }
    std::sort(found.begin(), found.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (auto& [t, ks] : found) {
        index_[ks.first] = {ks.second, ++clock_};
        bytes_ += ks.second;
    }
    {
        std::lock_guard<std::mutex> l(m_);
        evictLocked();
    }
    writer_ = std::thread([this] { writerLoop(); });
}

DiskCache::~DiskCache() {
    {
        std::lock_guard<std::mutex> l(m_);
        quit_ = true;
    }
    cv_.notify_all();
    if (writer_.joinable()) writer_.join();
}

std::string DiskCache::path(uint64_t k) const {
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)k);
    return (fs::path(dir_) / (std::string(b) + kExt)).string();
}

bool DiskCache::has(uint64_t k) {
    std::lock_guard<std::mutex> l(m_);
    if (index_.count(k)) return true;
    for (auto& q : queue_)
        if (q.first == k) return true;
    return current_.second && current_.first == k;
}

ImagePtr DiskCache::load(uint64_t k) {
    {
        std::lock_guard<std::mutex> l(m_);
        for (auto& q : queue_)  // still waiting to be written: hand it over directly
            if (q.first == k) return q.second;
        if (current_.second && current_.first == k) return current_.second;
        auto it = index_.find(k);
        if (it == index_.end()) return nullptr;
        it->second.use = ++clock_;
    }
    const std::string p = path(k);
    auto fail = [&]() -> ImagePtr {  // unreadable: forget it
        std::lock_guard<std::mutex> l(m_);
        auto it = index_.find(k);
        if (it != index_.end()) {
            bytes_ -= std::min(bytes_, it->second.bytes);
            index_.erase(it);
        }
        std::error_code ec;
        fs::remove(p, ec);
        return nullptr;
    };
    std::ifstream in(p, std::ios::binary);
    if (!in) return fail();
    char magic[4];
    int32_t box[4];
    uint32_t nch = 0;
    in.read(magic, 4);
    in.read((char*)box, sizeof box);
    in.read((char*)&nch, 4);
    if (!in || std::memcmp(magic, kMagic, 4) != 0 || nch > 4096) return fail();
    ChannelSet cs;
    for (uint32_t c = 0; c < nch; ++c) {
        uint16_t len = 0;
        in.read((char*)&len, 2);
        std::string name(len, '\0');
        in.read(name.data(), len);
        if (!in) return fail();
        cs.push_back(name);
    }
    Box b(box[0], box[1], box[2], box[3]);
    if (b.width() <= 0 || b.height() <= 0 || (size_t)b.area() > (size_t)1 << 28) return fail();
    auto img = std::make_shared<Image>(b, cs);
    const size_t n = img->data.size();
    std::vector<Imath::half> h(n);
    in.read((char*)h.data(), (std::streamsize)(n * sizeof(Imath::half)));
    if ((size_t)in.gcount() != n * sizeof(Imath::half)) return fail();
    float* d = img->data.data();
    for (size_t i = 0; i < n; ++i) d[i] = (float)h[i];
    in.close();
    std::error_code ec;  // recently used: keeps it across sessions too
    fs::last_write_time(p, fs::file_time_type::clock::now(), ec);
    return img;
}

void DiskCache::store(uint64_t k, const ImagePtr& image) {
    if (!image || image->data.empty()) return;
    {
        std::lock_guard<std::mutex> l(m_);
        if (budget_ == 0 || index_.count(k) || queue_.size() >= kMaxQueued) return;
        if (image->data.size() * sizeof(Imath::half) > budget_ / 4) return;  // one frame mustn't flush the rest
        for (auto& q : queue_)
            if (q.first == k) return;
        if (current_.second && current_.first == k) return;
        queue_.emplace_back(k, image);
    }
    cv_.notify_one();
}

void DiskCache::writerLoop() {
    for (;;) {
        std::pair<uint64_t, ImagePtr> job;
        {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait(l, [&] { return quit_ || !queue_.empty(); });
            if (quit_) return;  // unwritten frames are just not cached
            job = queue_.front();
            queue_.pop_front();
            current_ = job;
            writing_ = true;
        }
        const Image& im = *job.second;
        const std::string p = path(job.first), tmp = p + ".tmp";
        bool ok = false;
        size_t total = 0;
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (out) {
                int32_t box[4] = {im.bounds.x1, im.bounds.y1, im.bounds.x2, im.bounds.y2};
                uint32_t nch = (uint32_t)im.channels.size();
                out.write(kMagic, 4);
                out.write((const char*)box, sizeof box);
                out.write((const char*)&nch, 4);
                for (auto& c : im.channels) {
                    uint16_t len = (uint16_t)std::min<size_t>(c.size(), 65535);
                    out.write((const char*)&len, 2);
                    out.write(c.data(), len);
                }
                // in slices, so the conversion buffer stays small
                const size_t n = im.data.size(), slice = 1 << 20;
                std::vector<Imath::half> h(std::min(n, slice));
                for (size_t i = 0; i < n && out; i += slice) {
                    size_t m = std::min(slice, n - i);
                    for (size_t j = 0; j < m; ++j) h[j] = Imath::half(im.data[i + j]);
                    out.write((const char*)h.data(), (std::streamsize)(m * sizeof(Imath::half)));
                }
                ok = (bool)out;
                total = (size_t)out.tellp();
            }
        }
        std::error_code ec;
        if (ok) fs::rename(tmp, p, ec);
        if (!ok || ec) fs::remove(tmp, ec);
        {
            std::lock_guard<std::mutex> l(m_);
            current_ = {};
            if (ok && !ec && !index_.count(job.first) && !cleared_) {
                index_[job.first] = {total, ++clock_};
                bytes_ += total;
                evictLocked();
            } else if (ok && !ec && cleared_)
                fs::remove(p, ec);
            cleared_ = false;
            writing_ = false;
        }
        idle_.notify_all();
    }
}

void DiskCache::evictLocked() {
    while (bytes_ > budget_ && !index_.empty()) {
        auto old = std::min_element(index_.begin(), index_.end(), [](auto& a, auto& b) { return a.second.use < b.second.use; });
        std::error_code ec;
        fs::remove(path(old->first), ec);
        bytes_ -= std::min(bytes_, old->second.bytes);
        index_.erase(old);
    }
}

void DiskCache::clear() {
    std::unique_lock<std::mutex> l(m_);
    queue_.clear();
    if (writing_) cleared_ = true;  // the frame being written is dropped when it lands
    std::error_code ec;
    for (auto& [k, it] : index_) fs::remove(path(k), ec);
    index_.clear();
    bytes_ = 0;
}

void DiskCache::setBudget(size_t b) {
    std::lock_guard<std::mutex> l(m_);
    budget_ = b;
    evictLocked();
}

size_t DiskCache::bytes() {
    std::lock_guard<std::mutex> l(m_);
    return bytes_;
}

size_t DiskCache::budget() {
    std::lock_guard<std::mutex> l(m_);
    return budget_;
}

void DiskCache::flush() {
    std::unique_lock<std::mutex> l(m_);
    idle_.wait(l, [&] { return queue_.empty() && !writing_; });
}

}  // namespace strata
