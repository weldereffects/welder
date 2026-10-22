// Playback timing: decode-only and full render per frame (tests/playback_bench <comp> <node> [scale]).
#include <chrono>
#include <cstdio>
#include <fstream>
#include "core/Comp.h"
#include "core/Engine.h"
#include "io/ImageIO.h"
using namespace strata;

static double ms(std::chrono::steady_clock::time_point t0){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();}
int main(int argc,char**argv){
  registerBuiltinNodes();
  std::ifstream in(argv[1]); json j=json::parse(in);
  auto c=Comp::fromJson(j,"/tmp/claude-0/pb");
  double sc=argc>3?atof(argv[3]):0.5;
  Node* n=c->find(argv[2]);
  // decode only
  double dsum=0;
  for(int f=1001;f<1009;++f){auto t0=std::chrono::steady_clock::now(); ReadOptions o; auto r=readImage("/tmp/claude-0/pb/plate."+std::to_string(f)+".exr",o); dsum+=ms(t0);}
  std::printf("readImage only: %.1f ms/frame\n",dsum/8);
  Engine e; e.setScale(sc);
  double sum=0;
  for(int f=1009;f<1017;++f){auto t0=std::chrono::steady_clock::now(); e.clearMemo(); Box b=n->format(e,f).box(); e.renderRegion(n,f,b); double t=ms(t0); sum+=t; std::printf("frame %d %.1f ms\n",f,t);}
  std::printf("%s at scale %g: %.1f ms/frame (%.1f fps)\n",argv[2],sc,sum/8,8000/sum);
}
