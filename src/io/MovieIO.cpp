#include "../core/Crash.h"
#include "MovieIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include "../core/Node.h"

#ifdef STRATA_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#endif

namespace strata::movie {

namespace fs = std::filesystem;

bool isMoviePath(const std::string& path) {
    std::string e = fs::path(path).extension().string();
    for (auto& c : e) c = (char)std::tolower((unsigned char)c);
    return e == ".mov" || e == ".mp4" || e == ".m4v" || e == ".mkv" || e == ".avi" || e == ".mxf" || e == ".webm" ||
           e == ".mts" || e == ".m2ts" || e == ".ts" || e == ".mpg" || e == ".mpeg";
}

#ifndef STRATA_HAVE_FFMPEG

bool available() { return false; }
static RenderError noFFmpeg(const std::string& p) {
    return RenderError("movie files need FFmpeg, which this build of Strata doesn't include: " + p);
}
Info probe(const std::string& path) { throw noFFmpeg(path); }
std::shared_ptr<const raster::Raster> readFrame(const std::string& path, int) { throw noFFmpeg(path); }
struct Writer::Impl {};
Writer::Writer(const std::string& path, int, int, bool, const WriterOptions&) { throw noFFmpeg(path); }
Writer::~Writer() = default;
void Writer::write(const raster::Raster&) {}
void Writer::close() {}
int Writer::framesWritten() const { return 0; }

#else

bool available() { return true; }

namespace {

void quietLogs() {
    static std::once_flag once;
    std::call_once(once, [] { av_log_set_level(AV_LOG_ERROR); });
}

std::string averr(int e) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, buf, sizeof buf);
    return buf;
}

int fileKey(const std::string& path, std::string& key) {
    std::error_code ec;
    fs::path p = fs::path(std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size()));
    auto t = fs::last_write_time(p, ec);
    auto sz = fs::file_size(p, ec);
    key = path + "|" + std::to_string(ec ? 0 : (long long)t.time_since_epoch().count()) + "|" + std::to_string(ec ? 0 : sz);
    return 0;
}

const int* coefficientsFor(AVColorSpace cs, int height, bool jpeg) {
    int s;
    switch (cs) {
        case AVCOL_SPC_BT709: s = SWS_CS_ITU709; break;
        case AVCOL_SPC_BT470BG: case AVCOL_SPC_SMPTE170M: s = SWS_CS_ITU601; break;
        case AVCOL_SPC_SMPTE240M: s = SWS_CS_SMPTE240M; break;
        case AVCOL_SPC_FCC: s = SWS_CS_FCC; break;
        case AVCOL_SPC_BT2020_NCL: case AVCOL_SPC_BT2020_CL: s = SWS_CS_BT2020; break;
        default: s = jpeg || height < 720 ? SWS_CS_ITU601 : SWS_CS_ITU709;  // JFIF is BT.601
    }
    return sws_getCoefficients(s);
}

bool isFullRange(const AVFrame* f) {
    if (f->color_range == AVCOL_RANGE_JPEG) return true;
    switch (f->format) {
        case AV_PIX_FMT_YUVJ420P: case AV_PIX_FMT_YUVJ422P: case AV_PIX_FMT_YUVJ444P: case AV_PIX_FMT_YUVJ440P: return true;
        default: return false;
    }
}

struct Reader {
    std::mutex m;
    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws = nullptr;
    int si = -1;
    AVRational tb{1, 1}, rate{24, 1};
    int64_t start = 0;
    int nextIndex = -1;  // what the decoder returns next when read sequentially (-1: unknown)
    bool eof = false;
    Info info;
    std::string path;
    std::shared_ptr<const raster::Raster> last;  // the most recent decoded frame (for reads past the end)
    int lastIndex = -1;
    bool intraOnly = false;
    AVPixelFormat firstFormat = AV_PIX_FMT_NONE;  // of the first decoded frame
    std::vector<SwsContext*> bandSws;             // one per band, for threaded conversion

    explicit Reader(const std::string& p) : path(p) {
        quietLogs();
        int e = avformat_open_input(&fmt, p.c_str(), nullptr, nullptr);
        if (e < 0) throw RenderError("cannot open movie " + p + ": " + averr(e));
        // QuickTime / MP4 headers already hold the size, rate and frame count;
        // probing would decode a frame (single-threaded) only to learn the pixel
        // format, which the first real decode below tells us anyway.
        bool probed = false;
        auto headerKnows = [&] {
            if (std::strncmp(fmt->iformat->name, "mov", 3) != 0) return false;
            int v = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (v < 0) return false;
            const AVStream* s0 = fmt->streams[v];
            return s0->codecpar->width > 0 && s0->codecpar->height > 0 && (s0->avg_frame_rate.num > 0 || s0->r_frame_rate.num > 0);
        };
        if (!headerKnows()) {
            if ((e = avformat_find_stream_info(fmt, nullptr)) < 0) throw RenderError("cannot read movie " + p + ": " + averr(e));
            probed = true;
        }
        const AVCodec* codec = nullptr;
        si = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        if (si < 0) throw RenderError("no video stream in " + p);
        if (!codec) throw RenderError("no decoder in this build for the video in " + p);
        AVStream* st = fmt->streams[si];
        dec = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(dec, st->codecpar);
        dec->thread_count = 0;
        // Intra-only codecs (ProRes, DNxHR, ...) decode with slice threads: frame
        // threads must refill their pipeline after every seek, which makes
        // scrubbing slow. Long-GOP codecs keep frame threads.
        const AVCodecDescriptor* cd = avcodec_descriptor_get(codec->id);
        bool intra = cd && (cd->props & AV_CODEC_PROP_INTRA_ONLY);
        intraOnly = intra;
        dec->thread_type = intra && (codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) ? FF_THREAD_SLICE
                                                                                      : FF_THREAD_FRAME | FF_THREAD_SLICE;
        if ((e = avcodec_open2(dec, codec, nullptr)) < 0) throw RenderError("cannot decode " + p + ": " + averr(e));
        frame = av_frame_alloc();
        pkt = av_packet_alloc();
        tb = st->time_base;
        rate = st->avg_frame_rate.num && st->avg_frame_rate.den ? st->avg_frame_rate : st->r_frame_rate;
        if (!rate.num || !rate.den) rate = {24, 1};
        start = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;
        info.width = dec->width;
        info.height = dec->height;
        info.fps = av_q2d(rate);
        info.codec = codec->name;
        nextIndex = 0;
        if (dec->pix_fmt == AV_PIX_FMT_NONE) {
            // not probed: the first frame says (and is kept, so frame 1 is ready)
            try {
                get(0);
            } catch (...) {
                if (probed) throw;
                // fall back to the full probe
                avcodec_free_context(&dec);
                avformat_close_input(&fmt);
                if ((e = avformat_open_input(&fmt, p.c_str(), nullptr, nullptr)) < 0 ||
                    (e = avformat_find_stream_info(fmt, nullptr)) < 0)
                    throw RenderError("cannot read movie " + p + ": " + averr(e));
                dec = avcodec_alloc_context3(codec);
                avcodec_parameters_to_context(dec, fmt->streams[si]->codecpar);
                dec->thread_count = 0;
                if ((e = avcodec_open2(dec, codec, nullptr)) < 0) throw RenderError("cannot decode " + p + ": " + averr(e));
                st = fmt->streams[si];
                lastIndex = -1;
                eof = false;
            }
        }
        // (the decoder knows the profile once open, the header probe may not)
        int prof = st->codecpar->profile != AV_PROFILE_UNKNOWN ? st->codecpar->profile : dec->profile;
        if (const char* pn = avcodec_profile_name(st->codecpar->codec_id, prof)) info.profile = pn;
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(firstFormat != AV_PIX_FMT_NONE ? firstFormat : dec->pix_fmt);
        info.alpha = d && (d->flags & AV_PIX_FMT_FLAG_ALPHA);
        info.bits = d ? d->comp[0].depth : 8;
        if (info.width <= 0) {
            info.width = dec->width;
            info.height = dec->height;
        }
        AVRational sar = st->sample_aspect_ratio.num ? st->sample_aspect_ratio : dec->sample_aspect_ratio;
        info.aspect = sar.num && sar.den ? av_q2d(sar) : 1.0;
        if (st->nb_frames > 0) info.frames = (int)st->nb_frames;
        else if (st->duration != AV_NOPTS_VALUE) info.frames = (int)std::llround(st->duration * av_q2d(tb) * info.fps);
        else if (fmt->duration != AV_NOPTS_VALUE) info.frames = (int)std::llround(fmt->duration / (double)AV_TIME_BASE * info.fps);
    }
    ~Reader() {
        for (auto* b : bandSws) sws_freeContext(b);
        sws_freeContext(sws);
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
    }

    int indexOf(int64_t pts) const { return (int)std::llround((pts - start) * av_q2d(tb) * info.fps); }
    int64_t ptsOf(int index) const { return start + (int64_t)std::llround(index / info.fps / av_q2d(tb)); }

    bool decodeNext() {
        while (true) {
            int e = avcodec_receive_frame(dec, frame);
            if (e == 0) return true;
            if (e == AVERROR_EOF) return false;
            if (e != AVERROR(EAGAIN)) throw RenderError("decoding " + path + ": " + averr(e));
            if (eof) {
                avcodec_send_packet(dec, nullptr);
                continue;
            }
            e = av_read_frame(fmt, pkt);
            if (e < 0) {
                eof = true;
                avcodec_send_packet(dec, nullptr);
                continue;
            }
            if (pkt->stream_index == si) avcodec_send_packet(dec, pkt);
            av_packet_unref(pkt);
        }
    }

    void seek(int index) {
        av_seek_frame(fmt, si, ptsOf(index), AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(dec);
        eof = false;
        nextIndex = -1;
    }

    std::shared_ptr<const raster::Raster> get(int index) {
        if (index < 0) index = 0;
        if (index == lastIndex) return last;
        // intra-only codecs seek straight to any frame; others decode up to 24 ahead
        if (nextIndex < 0 || index < nextIndex || index > nextIndex + (intraOnly ? 0 : 24)) seek(index);
        bool sought = nextIndex < 0;
        while (true) {
            if (!decodeNext()) {
                if (lastIndex >= 0) return last;  // past the end: hold the last frame
                if (!sought) { seek(0); sought = true; continue; }
                throw RenderError("no frame " + std::to_string(index + 1) + " in " + path);
            }
            int64_t ts = frame->best_effort_timestamp != AV_NOPTS_VALUE ? frame->best_effort_timestamp : frame->pts;
            int fi = ts == AV_NOPTS_VALUE ? (nextIndex < 0 ? index : nextIndex) : indexOf(ts);
            nextIndex = fi + 1;
            if (fi < index) {
                av_frame_unref(frame);
                continue;
            }
            last = std::make_shared<const raster::Raster>(convert());
            lastIndex = index;
            av_frame_unref(frame);
            return last;
        }
    }

    raster::Raster convert() {
        const int W = frame->width, H = frame->height;
        if (firstFormat == AV_PIX_FMT_NONE) firstFormat = (AVPixelFormat)frame->format;
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((AVPixelFormat)frame->format);
        bool alpha = d && (d->flags & AV_PIX_FMT_FLAG_ALPHA);
        AVPixelFormat outFmt = alpha ? AV_PIX_FMT_RGBA64 : AV_PIX_FMT_RGB48;
        const int flags = SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT | SWS_FULL_CHR_H_INP;
        const bool yuv = d && !(d->flags & AV_PIX_FMT_FLAG_RGB);
        const int* coef = yuv ? coefficientsFor(frame->colorspace, H, dec->codec_id == AV_CODEC_ID_MJPEG) : nullptr;
        bool grey = d && d->nb_components <= 2;  // grey images: full range unless tagged otherwise
        bool full = isFullRange(frame) || (grey && frame->color_range != AVCOL_RANGE_MPEG);
        const int n = alpha ? 4 : 3;
        raster::Raster r;
        r.w = W; r.h = H; r.n = n; r.bits = d ? d->comp[0].depth : info.bits;
        r.px.resize((size_t)W * H * n);
        // Rows are independent when chroma isn't subsampled vertically (4:4:4,
        // 4:2:2, RGB: ProRes, DNx...), so bands convert on all cores at once.
        int bands = 1;
        if (d && d->log2_chroma_h == 0 && !(d->flags & AV_PIX_FMT_FLAG_PAL) && !(d->flags & AV_PIX_FMT_FLAG_BITSTREAM))
            bands = std::clamp((int)std::thread::hardware_concurrency(), 1, std::max(1, H / 64));
        if ((int)bandSws.size() < bands) bandSws.resize(bands, nullptr);
        std::vector<std::string> errs(bands);
        auto work = [&](int b) {
            const int y0 = (int)((int64_t)H * b / bands), y1 = (int)((int64_t)H * (b + 1) / bands), bh = y1 - y0;
            SwsContext*& ctx = bands == 1 ? sws : bandSws[b];
            ctx = sws_getCachedContext(ctx, W, bh, (AVPixelFormat)frame->format, W, bh, outFmt, flags, nullptr, nullptr, nullptr);
            if (!ctx) {
                errs[b] = "cannot convert the pixels of " + path;
                return;
            }
            if (yuv) sws_setColorspaceDetails(ctx, coef, full ? 1 : 0, coef, 1, 0, 1 << 16, 1 << 16);
            const uint8_t* src[4] = {nullptr, nullptr, nullptr, nullptr};
            for (int pl = 0; pl < 4 && frame->data[pl]; ++pl)
                src[pl] = frame->data[pl] + (ptrdiff_t)y0 * frame->linesize[pl];
            std::vector<uint16_t> buf((size_t)W * bh * n);
            uint8_t* dst[4] = {(uint8_t*)buf.data(), nullptr, nullptr, nullptr};
            int stride[4] = {W * n * 2, 0, 0, 0};
            sws_scale(ctx, src, frame->linesize, 0, bh, dst, stride);
            float* o = r.px.data() + (size_t)y0 * W * n;
            const float k = 1.0f / 65535.0f;
            for (size_t i = 0; i < buf.size(); ++i) o[i] = buf[i] * k;
        };
        if (bands == 1) work(0);
        else {
            std::vector<std::thread> ts;
            for (int b = 1; b < bands; ++b) ts.emplace_back(work, b);
            work(0);
            for (auto& t : ts) t.join();
        }
        for (auto& e : errs)
            if (!e.empty()) throw RenderError(e);
        return r;
    }
};

std::mutex gReadersM;
std::list<std::pair<std::string, std::shared_ptr<Reader>>> gReaders;

std::shared_ptr<Reader> reader(const std::string& path) {
    std::string key;
    fileKey(path, key);
    std::lock_guard<std::mutex> l(gReadersM);
    for (auto it = gReaders.begin(); it != gReaders.end(); ++it)
        if (it->first == key) {
            gReaders.splice(gReaders.begin(), gReaders, it);
            return gReaders.front().second;
        }
    auto r = std::make_shared<Reader>(path);
    gReaders.emplace_front(key, r);
    while (gReaders.size() > 6) gReaders.pop_back();
    return r;
}

// A movie Strata (re)writes: drop its decoders, whatever its timestamp says.
void forget(const std::string& path) {
    std::lock_guard<std::mutex> l(gReadersM);
    gReaders.remove_if([&](auto& r) { return r.second->path == path; });
}

}  // namespace

Info probe(const std::string& path) {
    auto r = reader(path);
    std::lock_guard<std::mutex> l(r->m);
    return r->info;
}

std::shared_ptr<const raster::Raster> readFrame(const std::string& path, int index) {
    crash::crumb("read movie frame " + std::to_string(index) + " of " + path);
    auto r = reader(path);
    std::lock_guard<std::mutex> l(r->m);
    return r->get(index);
}

// ------------------------------------------------------------------ Writer

struct Writer::Impl {
    std::string path;
    AVFormatContext* oc = nullptr;
    AVCodecContext* enc = nullptr;
    AVStream* st = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws = nullptr;
    bool alpha = false, open = false, rgbOut = false, fullRange = false;
    int64_t next = 0;
    int qscale = 0;

    ~Impl() {
        sws_freeContext(sws);
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avcodec_free_context(&enc);
        if (oc) {
            if (oc->pb && !(oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&oc->pb);
            avformat_free_context(oc);
        }
    }

    void drain(bool flush) {
        if (flush) avcodec_send_frame(enc, nullptr);
        while (true) {
            int e = avcodec_receive_packet(enc, pkt);
            if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) return;
            if (e < 0) throw RenderError("encoding " + path + ": " + averr(e));
            av_packet_rescale_ts(pkt, enc->time_base, st->time_base);
            pkt->stream_index = st->index;
            e = av_interleaved_write_frame(oc, pkt);
            if (e < 0) throw RenderError("writing " + path + ": " + averr(e));
        }
    }
};

Writer::Writer(const std::string& path, int W, int H, bool alpha, const WriterOptions& o) : d_(std::make_unique<Impl>()) {
    quietLogs();
    forget(path);
    Impl& d = *d_;
    d.path = path;
    std::string c = o.codec;
    std::string encName;
    AVPixelFormat pf = AV_PIX_FMT_YUV422P10;
    AVDictionary* opts = nullptr;
    int profile = -1;
    if (c.rfind("prores", 0) == 0) {
        encName = "prores_ks";
        profile = c == "prores 422 proxy" ? 0 : c == "prores 422 lt" ? 1 : c == "prores 422" ? 2 : c == "prores 422 hq" ? 3
                : c == "prores 4444" ? 4 : c == "prores 4444 xq" ? 5 : 3;
        bool p4444 = profile >= 4;
        d.alpha = alpha && p4444;
        pf = p4444 ? (d.alpha ? AV_PIX_FMT_YUVA444P10 : AV_PIX_FMT_YUV444P10) : AV_PIX_FMT_YUV422P10;
        const char* names[] = {"proxy", "lt", "standard", "hq", "4444", "4444xq"};
        av_dict_set(&opts, "profile", names[profile], 0);  // prores_ks takes it as its own option
        av_dict_set(&opts, "vendor", "apl0", 0);
        if (d.alpha) av_dict_set(&opts, "alpha_bits", "16", 0);
    } else if (c.rfind("dnxhr", 0) == 0) {
        encName = "dnxhd";
        std::string prof = c == "dnxhr 444" ? "dnxhr_444" : c == "dnxhr hqx" ? "dnxhr_hqx" : c == "dnxhr hq" ? "dnxhr_hq"
                         : c == "dnxhr lb" ? "dnxhr_lb" : "dnxhr_sq";
        av_dict_set(&opts, "profile", prof.c_str(), 0);
        pf = prof == "dnxhr_444" ? AV_PIX_FMT_YUV444P10 : prof == "dnxhr_hqx" ? AV_PIX_FMT_YUV422P10 : AV_PIX_FMT_YUV422P;
    } else if (c == "mjpeg") {
        encName = "mjpeg";
        pf = AV_PIX_FMT_YUVJ422P;
        d.fullRange = true;
    } else if (c == "mpeg4") {
        encName = "mpeg4";
        pf = AV_PIX_FMT_YUV420P;
    } else if (c == "ffv1") {
        encName = "ffv1";
        d.alpha = alpha;
        pf = alpha ? AV_PIX_FMT_GBRAP16 : AV_PIX_FMT_GBRP16;
        d.rgbOut = true;
        av_dict_set(&opts, "level", "3", 0);
        av_dict_set(&opts, "slicecrc", "1", 0);
    } else if (c == "png") {
        encName = "png";
        d.alpha = alpha;
        pf = alpha ? AV_PIX_FMT_RGBA64BE : AV_PIX_FMT_RGB48BE;
        d.rgbOut = true;
    } else {
        throw RenderError("unknown movie codec \"" + c + "\"");
    }
    const AVCodec* codec = avcodec_find_encoder_by_name(encName.c_str());
    if (!codec) throw RenderError("this build has no " + encName + " encoder");
    int e = avformat_alloc_output_context2(&d.oc, nullptr, nullptr, path.c_str());
    if (e < 0 || !d.oc) throw RenderError("no movie format for " + path + " (use .mov .mp4 .mkv .avi .mxf)");
    if (avformat_query_codec(d.oc->oformat, codec->id, FF_COMPLIANCE_NORMAL) == 0)
        throw RenderError(std::string("a ") + d.oc->oformat->name + " file can't hold " + c + "; use .mov or .mkv");
    d.enc = avcodec_alloc_context3(codec);
    AVRational rate = av_d2q(o.fps > 0 ? o.fps : 24, 100000);
    d.enc->width = W;
    d.enc->height = H;
    d.enc->time_base = av_inv_q(rate);
    d.enc->framerate = rate;
    d.enc->pix_fmt = pf;
    d.enc->sample_aspect_ratio = {1, 1};
    d.enc->thread_count = 0;
    if (profile >= 0) d.enc->profile = profile;
    if (!d.rgbOut) {
        d.enc->color_primaries = AVCOL_PRI_BT709;
        d.enc->color_trc = AVCOL_TRC_BT709;
        d.enc->colorspace = d.fullRange ? AVCOL_SPC_BT470BG : AVCOL_SPC_BT709;  // MJPEG: JFIF's BT.601 matrix
        d.enc->color_range = d.fullRange ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    }
    if (c == "mjpeg" || c == "mpeg4") {
        int q = std::clamp(o.quality, 1, 100);
        d.qscale = std::clamp(2 + (100 - q) * 29 / 99, 1, 31);
        d.enc->flags |= AV_CODEC_FLAG_QSCALE;
        d.enc->global_quality = FF_QP2LAMBDA * d.qscale;
        if (c == "mpeg4") d.enc->gop_size = 12;
    }
    if (d.oc->oformat->flags & AVFMT_GLOBALHEADER) d.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (encName == "dnxhd" && (W < 256 || H < 120)) {
        av_dict_free(&opts);
        throw RenderError("DNxHR needs at least 256x120 pixels (this is " + std::to_string(W) + "x" + std::to_string(H) + ")");
    }
    if ((pf == AV_PIX_FMT_YUV420P || pf == AV_PIX_FMT_YUV422P || pf == AV_PIX_FMT_YUV422P10 || pf == AV_PIX_FMT_YUVJ422P) &&
        (W % 2 || (pf == AV_PIX_FMT_YUV420P && H % 2))) {
        av_dict_free(&opts);
        throw RenderError(c + " needs an even width" + std::string(pf == AV_PIX_FMT_YUV420P ? " and height" : "") +
                          " (this is " + std::to_string(W) + "x" + std::to_string(H) + "); try prores 4444 or ffv1");
    }
    if ((e = avcodec_open2(d.enc, codec, &opts)) < 0) {
        av_dict_free(&opts);
        throw RenderError("cannot start the " + c + " encoder for " + std::to_string(W) + "x" + std::to_string(H) + ": " + averr(e));
    }
    av_dict_free(&opts);
    d.st = avformat_new_stream(d.oc, nullptr);
    avcodec_parameters_from_context(d.st->codecpar, d.enc);
    d.st->time_base = d.enc->time_base;
    d.st->avg_frame_rate = rate;
    if (!(d.oc->oformat->flags & AVFMT_NOFILE) && (e = avio_open(&d.oc->pb, path.c_str(), AVIO_FLAG_WRITE)) < 0)
        throw RenderError("cannot write " + path + ": " + averr(e));
    if ((e = avformat_write_header(d.oc, nullptr)) < 0) throw RenderError("cannot write " + path + ": " + averr(e));
    d.frame = av_frame_alloc();
    d.frame->format = pf;
    d.frame->width = W;
    d.frame->height = H;
    if ((e = av_frame_get_buffer(d.frame, 0)) < 0) throw RenderError("out of memory: " + averr(e));
    d.pkt = av_packet_alloc();
    d.open = true;
}

Writer::~Writer() {
    try {
        close();
    } catch (...) {
    }
}

void Writer::write(const raster::Raster& r) {
    Impl& d = *d_;
    if (!d.open) throw RenderError(d.path + " is already closed");
    const int W = d.enc->width, H = d.enc->height;
    if (r.w != W || r.h != H)
        throw RenderError("movie frames must all be " + std::to_string(W) + "x" + std::to_string(H) + " (got " +
                          std::to_string(r.w) + "x" + std::to_string(r.h) + ")");
    int n = d.alpha ? 4 : 3;
    std::vector<uint16_t> buf((size_t)W * H * n);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float* s = r.at(x, y);
            uint16_t* o = &buf[((size_t)y * W + x) * n];
            for (int c = 0; c < n; ++c) {
                float v = r.n >= 3 ? (c < r.n ? s[c] : 1.0f) : (c == 3 ? (r.n == 2 ? s[1] : 1.0f) : s[0]);
                o[c] = (uint16_t)std::lround(std::clamp(v, 0.0f, 1.0f) * 65535.0f);
            }
        }
    AVPixelFormat inFmt = d.alpha ? AV_PIX_FMT_RGBA64 : AV_PIX_FMT_RGB48;
    d.sws = sws_getCachedContext(d.sws, W, H, inFmt, W, H, d.enc->pix_fmt, SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INP,
                                 nullptr, nullptr, nullptr);
    if (!d.sws) throw RenderError("cannot convert pixels for " + d.path);
    if (!d.rgbOut) {
        const int* cs = sws_getCoefficients(d.fullRange ? SWS_CS_ITU601 : SWS_CS_ITU709);
        sws_setColorspaceDetails(d.sws, cs, 1, cs, d.fullRange ? 1 : 0, 0, 1 << 16, 1 << 16);
    }
    int e = av_frame_make_writable(d.frame);
    if (e < 0) throw RenderError("encoding " + d.path + ": " + averr(e));
    const uint8_t* src[4] = {(const uint8_t*)buf.data(), nullptr, nullptr, nullptr};
    int stride[4] = {W * n * 2, 0, 0, 0};
    sws_scale(d.sws, src, stride, 0, H, d.frame->data, d.frame->linesize);
    d.frame->pts = d.next++;
    if (d.qscale) d.frame->quality = FF_QP2LAMBDA * d.qscale;
    if ((e = avcodec_send_frame(d.enc, d.frame)) < 0) throw RenderError("encoding " + d.path + ": " + averr(e));
    d.drain(false);
}

void Writer::close() {
    Impl& d = *d_;
    if (!d.open) return;
    d.open = false;
    d.drain(true);
    int e = av_write_trailer(d.oc);
    if (d.oc->pb && !(d.oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&d.oc->pb);
    forget(d.path);
    if (e < 0) throw RenderError("finishing " + d.path + ": " + averr(e));
}

int Writer::framesWritten() const { return (int)d_->next; }

#endif

}  // namespace strata::movie
