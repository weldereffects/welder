#!/usr/bin/env bash
# Build a small static FFmpeg (LGPL, no external codecs) for Welder's movie I/O.
#
#   tools/build_ffmpeg.sh <src-dir> <install-dir> [--windows] [<zlib-prefix>]
#
# Reads ProRes, H.264, HEVC, DNxHD/HR, MJPEG, MPEG-2/4, VP9, FFV1, PNG, QuickTime
# RLE, v210, DV and raw video in MOV / MP4 / MKV / AVI / MXF / TS; writes ProRes,
# DNxHD/HR, MJPEG, MPEG-4, FFV1, PNG and raw video, and H.264 / H.265 through the
# hardware encoders the machine has: NVIDIA NVENC, AMD AMF, Vulkan Video and (Windows)
# Media Foundation. Each of those loads its driver at run time, so nothing extra is
# linked or shipped. (libx264 / libx265 are GPL, so they are left out.)
# Then configure Welder with -DSTRATA_FFMPEG_DIR=<install-dir>.
set -euo pipefail
SRC=$(mkdir -p "$1" && cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
WIN=${3:-}
ZLIB=${4:-}
TAG=n7.1.1
[ -d "$SRC/FFmpeg" ] || git clone -q --depth 1 --branch $TAG https://github.com/FFmpeg/FFmpeg.git "$SRC/FFmpeg"

# Headers only, for the hardware H.264 / H.265 encoders (all MIT / Apache-2.0; the drivers are
# loaded at run time). Kept out of $OUT/include so Welder's own Vulkan headers aren't shadowed.
HW="$OUT/hw-headers"
if [ ! -f "$HW/lib/pkgconfig/ffnvcodec.pc" ]; then  # NVENC: needs driver 531+ (Windows) / 530+ (Linux)
  [ -d "$SRC/nv-codec-headers" ] || git clone -q --depth 1 --branch n12.1.14.0 https://github.com/FFmpeg/nv-codec-headers.git "$SRC/nv-codec-headers"
  make -C "$SRC/nv-codec-headers" PREFIX="$HW" install > /dev/null
fi
if [ ! -d "$HW/include/AMF" ]; then
  [ -d "$SRC/AMF" ] || { git clone -q --depth 1 --branch v1.4.36 --filter=blob:none --sparse https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git "$SRC/AMF" &&
                         git -C "$SRC/AMF" sparse-checkout set amf/public/include; }
  mkdir -p "$HW/include" && cp -r "$SRC/AMF/amf/public/include" "$HW/include/AMF"
fi
if [ ! -f "$HW/include/vulkan/vulkan.h" ]; then  # Vulkan Video encode wants headers 1.3.277+
  [ -d "$SRC/Vulkan-Headers" ] || git clone -q --depth 1 --branch v1.3.296 https://github.com/KhronosGroup/Vulkan-Headers.git "$SRC/Vulkan-Headers"
  mkdir -p "$HW/include" && cp -r "$SRC/Vulkan-Headers/include/vulkan" "$SRC/Vulkan-Headers/include/vk_video" "$HW/include/"
fi
export PKG_CONFIG_PATH="$HW/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

DEC=prores,h264,hevc,dnxhd,mjpeg,mpeg4,mpeg2video,mpeg1video,vp9,ffv1,png,qtrle,v210,v210x,dvvideo,rawvideo,tiff,dpx,exr
ENC=prores_ks,dnxhd,mjpeg,mpeg4,ffv1,png,rawvideo,v210
HWENC=h264_nvenc,hevc_nvenc,h264_amf,hevc_amf,h264_vulkan,hevc_vulkan  # H.264 / H.265: hardware only
HWARGS=(--enable-ffnvcodec --enable-nvenc --enable-amf --enable-vulkan)
DMX=mov,matroska,avi,mxf,mpegts,mpegps,m4v,h264,hevc,rawvideo,image2
MUX=mov,mp4,matroska,avi,mxf
PAR=h264,hevc,mpeg4video,mpegvideo,mjpeg,dnxhd,vp9,png,dvd_nav,prores
BSF=h264_mp4toannexb,hevc_mp4toannexb,vp9_superframe_split

ARGS=(--prefix="$OUT" --disable-everything --enable-static --disable-shared --enable-pic
      --disable-programs --disable-doc --disable-network --disable-autodetect --disable-debug
      --disable-avdevice --disable-avfilter --disable-swresample --disable-postproc
      --enable-avcodec --enable-avformat --enable-swscale --enable-avutil
      --enable-decoder=$DEC --enable-encoder=$ENC --enable-demuxer=$DMX --enable-muxer=$MUX
      --enable-parser=$PAR --enable-bsf=$BSF --enable-protocol=file --enable-zlib
      --pkg-config=pkg-config --extra-cflags="-I$HW/include")
if [ "$WIN" = "--windows" ]; then
  ARGS+=(--target-os=mingw32 --arch=x86_64 --enable-cross-compile --cross-prefix=x86_64-w64-mingw32-
         --cc=x86_64-w64-mingw32-gcc-posix --cxx=x86_64-w64-mingw32-g++-posix --enable-w32threads
         "${HWARGS[@]}" --enable-mediafoundation --enable-encoder=$HWENC,h264_mf,hevc_mf)
else
  ARGS+=(--enable-pthreads)
  if [ "$(uname)" != Darwin ]; then  # Linux: NVENC, AMF and Vulkan Video (Mac: none of them yet)
    ARGS+=("${HWARGS[@]}" --enable-encoder=$HWENC)
  fi
  if [ "$(uname)" = Darwin ]; then  # nothing from Apple's frameworks: Welder links only what it names
    if [ "${WELDER_MAC_ARCH:-$(uname -m)}" != "$(uname -m)" ]; then  # arm64 on an Intel Mac
      ARGS+=(--enable-cross-compile --target-os=darwin --arch=aarch64 --cc="clang -arch arm64" --cxx="clang++ -arch arm64")
    fi
    ARGS+=(--disable-iconv --disable-securetransport --disable-videotoolbox --disable-audiotoolbox
           --disable-avfoundation --disable-coreimage --disable-appkit --disable-metal)
  fi
fi
if [ -n "$ZLIB" ]; then
  ARGS+=(--extra-cflags="-I$ZLIB/include" --extra-ldflags="-L$ZLIB/lib")
fi
B="$OUT/build"
mkdir -p "$B"
cd "$B"
"$SRC/FFmpeg/configure" "${ARGS[@]}" > configure.log 2>&1 || { tail -30 configure.log; tail -30 ffbuild/config.log; exit 1; }
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" > make.log 2>&1 || { tail -40 make.log; exit 1; }
make install > install.log 2>&1
# say which H.264 / H.265 encoders made it in (a missing header quietly drops one)
for e in h264_nvenc hevc_nvenc h264_amf hevc_amf h264_vulkan hevc_vulkan h264_mf hevc_mf; do
  if grep -q "^#define CONFIG_$(echo $e | tr a-z A-Z)_ENCODER 1" config_components.h; then echo "  encoder: $e"; fi
done
echo "FFmpeg installed in $OUT"
