#!/usr/bin/env bash
# Build a small static FFmpeg (LGPL, no external codecs) for Welder's movie I/O.
#
#   tools/build_ffmpeg.sh <src-dir> <install-dir> [--windows] [<zlib-prefix>]
#
# Reads ProRes, H.264, HEVC, DNxHD/HR, MJPEG, MPEG-2/4, VP9, FFV1, PNG, QuickTime
# RLE, v210, DV and raw video in MOV / MP4 / MKV / AVI / MXF / TS; writes ProRes,
# DNxHD/HR, MJPEG, MPEG-4, FFV1, PNG and raw video. (H.264 encoding needs libx264,
# which is GPL, so it is left out.) Then configure Welder with
# -DSTRATA_FFMPEG_DIR=<install-dir>.
set -euo pipefail
SRC=$(mkdir -p "$1" && cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
WIN=${3:-}
ZLIB=${4:-}
TAG=n7.1.1
[ -d "$SRC/FFmpeg" ] || git clone -q --depth 1 --branch $TAG https://github.com/FFmpeg/FFmpeg.git "$SRC/FFmpeg"

DEC=prores,h264,hevc,dnxhd,mjpeg,mpeg4,mpeg2video,mpeg1video,vp9,ffv1,png,qtrle,v210,v210x,dvvideo,rawvideo,tiff,dpx,exr
ENC=prores_ks,dnxhd,mjpeg,mpeg4,ffv1,png,rawvideo,v210
DMX=mov,matroska,avi,mxf,mpegts,mpegps,m4v,h264,hevc,rawvideo,image2
MUX=mov,mp4,matroska,avi,mxf
PAR=h264,hevc,mpeg4video,mpegvideo,mjpeg,dnxhd,vp9,png,dvd_nav,prores
BSF=h264_mp4toannexb,hevc_mp4toannexb,vp9_superframe_split

ARGS=(--prefix="$OUT" --disable-everything --enable-static --disable-shared --enable-pic
      --disable-programs --disable-doc --disable-network --disable-autodetect --disable-debug
      --disable-avdevice --disable-avfilter --disable-swresample --disable-postproc
      --enable-avcodec --enable-avformat --enable-swscale --enable-avutil
      --enable-decoder=$DEC --enable-encoder=$ENC --enable-demuxer=$DMX --enable-muxer=$MUX
      --enable-parser=$PAR --enable-bsf=$BSF --enable-protocol=file --enable-zlib)
if [ "$WIN" = "--windows" ]; then
  ARGS+=(--target-os=mingw32 --arch=x86_64 --enable-cross-compile --cross-prefix=x86_64-w64-mingw32-
         --cc=x86_64-w64-mingw32-gcc-posix --cxx=x86_64-w64-mingw32-g++-posix --enable-w32threads)
else
  ARGS+=(--enable-pthreads)
  if [ "$(uname)" = Darwin ]; then  # nothing from Apple's frameworks: Welder links only what it names
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
echo "FFmpeg installed in $OUT"
