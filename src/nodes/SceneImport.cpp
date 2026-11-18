// Camera + point cloud import for Scene3D (SceneImport.h).
#include "SceneImport.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "../core/CacheDir.h"
#include "../core/Process.h"
#include "BlenderNode.h"
#include "ufbx/ufbx.h"

namespace strata::scene3d {

using cam3d::M3;
using cam3d::V3;

namespace {

std::string lower(std::string s) {
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

V3 axisVec(ufbx_coordinate_axis a) {
    switch (a) {
        case UFBX_COORDINATE_AXIS_POSITIVE_X: return V3(1, 0, 0);
        case UFBX_COORDINATE_AXIS_NEGATIVE_X: return V3(-1, 0, 0);
        case UFBX_COORDINATE_AXIS_POSITIVE_Y: return V3(0, 1, 0);
        case UFBX_COORDINATE_AXIS_NEGATIVE_Y: return V3(0, -1, 0);
        case UFBX_COORDINATE_AXIS_POSITIVE_Z: return V3(0, 0, 1);
        case UFBX_COORDINATE_AXIS_NEGATIVE_Z: return V3(0, 0, -1);
        default: return V3();
    }
}

V3 dirOf(const ufbx_matrix& M, const V3& d) {
    ufbx_vec3 r = ufbx_transform_direction(&M, ufbx_vec3{d.x, d.y, d.z});
    return V3(r.x, r.y, r.z);
}
V3 posOf(const ufbx_matrix& M) { return V3(M.cols[3].x, M.cols[3].y, M.cols[3].z); }

// The camera on `node` in `scene` (possibly an evaluated copy), in Welder's terms.
CameraSample sampleCamera(const ufbx_node* node, double frame) {
    CameraSample s;
    s.frame = frame;
    const ufbx_camera* cam = node->camera;
    const ufbx_matrix& M = node->geometry_to_world;
    s.C = posOf(M);
    // the camera's own axes: right, up and "front" (towards the viewer: Welder's camera +Z)
    // (FBX's own camera space when ufbx reports none: looks down +X, +Y up, so right = +Z, front = -X)
    const ufbx_coordinate_axes ax = cam->projection_axes;
    const bool valid = ufbx_coordinate_axes_valid(ax);
    V3 x = cam3d::normalize(dirOf(M, valid ? axisVec(ax.right) : V3(0, 0, 1)));
    V3 y = cam3d::normalize(dirOf(M, valid ? axisVec(ax.up) : V3(0, 1, 0)));
    V3 z = cam3d::normalize(dirOf(M, valid ? axisVec(ax.front) : V3(-1, 0, 0)));
    for (int i = 0; i < 3; ++i) {
        s.R.m[i][0] = x[i];
        s.R.m[i][1] = y[i];
        s.R.m[i][2] = z[i];
    }
    s.R = cam3d::orthonormalize(s.R);
    if (cam->focal_length_mm > 0) s.focal = cam->focal_length_mm;
    ufbx_vec2 film = cam->film_size_inch;
    if (!(film.x > 0 && film.y > 0)) film = cam->aperture_size_inch;
    if (film.x > 0 && film.y > 0) {
        s.haperture = film.x * 25.4;
        s.vaperture = film.y * 25.4;
    }
    return s;
}

}  // namespace

// ================================================================= FBX
bool readFbx(const std::string& path, double fps, ImportedScene& out, std::string* err) {
    ufbx_load_opts opts = {};
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.space_conversion = UFBX_SPACE_CONVERSION_ADJUST_TRANSFORMS;
    ufbx_error uerr;
    ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &uerr);
    if (!scene) {
        if (err) *err = "can't read " + path + ": " + std::string(uerr.description.data, uerr.description.length);
        return false;
    }
    struct Free {
        ufbx_scene* s;
        ~Free() { ufbx_free_scene(s); }
    } freeScene{scene};

    if (scene->settings.frames_per_second > 0) fps = scene->settings.frames_per_second;
    if (!(fps > 0)) fps = 24;
    out.fps = fps;
    const ufbx_anim* anim = scene->anim;
    double t0 = anim ? anim->time_begin : 0, t1 = anim ? anim->time_end : 0;
    if (t1 <= t0 && scene->anim_curves.count) {  // no range on the take: the keys' range
        t0 = 1e300, t1 = -1e300;
        for (size_t i = 0; i < scene->anim_curves.count; ++i) {
            const ufbx_anim_curve* c = scene->anim_curves.data[i];
            if (!c->keyframes.count) continue;
            t0 = std::min(t0, c->min_time);
            t1 = std::max(t1, c->max_time);
        }
        if (t1 < t0) t0 = t1 = 0;
    }
    long f0 = std::lround(t0 * fps), f1 = std::lround(t1 * fps);
    if (f1 < f0) f1 = f0;
    if (f1 - f0 > 100000) f1 = f0 + 100000;

    // the camera: of the cameras in the file, the one that moves the most
    std::vector<const ufbx_node*> cams;
    for (size_t i = 0; i < scene->nodes.count; ++i)
        if (scene->nodes.data[i]->camera) cams.push_back(scene->nodes.data[i]);
    if (cams.empty()) {
        if (err) *err = path + " has no camera";
        return false;
    }
    auto evalAt = [&](double t) -> ufbx_scene* {
        if (!anim) return nullptr;
        ufbx_error e;
        return ufbx_evaluate_scene(scene, anim, t, nullptr, &e);
    };
    const ufbx_node* best = cams[0];
    if (cams.size() > 1 && anim && f1 > f0) {
        ufbx_scene* a = evalAt(f0 / fps);
        ufbx_scene* b = evalAt(f1 / fps);
        double bestMove = -1;
        if (a && b)
            for (const ufbx_node* n : cams) {
                double mv = cam3d::norm(posOf(a->nodes.data[n->typed_id]->node_to_world) - posOf(b->nodes.data[n->typed_id]->node_to_world));
                if (mv > bestMove + 1e-9) bestMove = mv, best = n;
            }
        if (a) ufbx_free_scene(a);
        if (b) ufbx_free_scene(b);
    }
    out.cameraName.assign(best->name.data, best->name.length);
    out.nearClip = best->camera->near_plane > 0 ? best->camera->near_plane : 0.1;
    out.farClip = best->camera->far_plane > out.nearClip ? best->camera->far_plane : 10000;
    if (!anim || f1 == f0) {
        out.samples.push_back(sampleCamera(best, (double)f0));
    } else {
        for (long f = f0; f <= f1; ++f) {
            ufbx_scene* ev = evalAt(f / fps);
            if (!ev) continue;
            out.samples.push_back(sampleCamera(ev->nodes.data[best->typed_id], (double)f));
            ufbx_free_scene(ev);
        }
    }
    if (out.samples.empty()) {
        if (err) *err = path + ": the camera couldn't be evaluated";
        return false;
    }
    // points: leaf locators / nulls, and vertex-only meshes (not the camera's own rig)
    std::vector<char> rig(scene->nodes.count, 0);
    for (const ufbx_node* n = best; n; n = n->parent) rig[n->typed_id] = 1;
    for (size_t i = 0; i < scene->nodes.count; ++i) {
        const ufbx_node* n = scene->nodes.data[i];
        if (n->is_root || rig[n->typed_id] || n->camera || n->light) continue;
        if (n->mesh) {
            const ufbx_mesh* m = n->mesh;
            if (m->num_faces > 0) continue;
            for (size_t v = 0; v < m->vertices.count; ++v) {
                ufbx_vec3 p = ufbx_transform_position(&n->geometry_to_world, m->vertices.data[v]);
                out.points.emplace_back(p.x, p.y, p.z);
            }
            continue;
        }
        if (n->children.count == 0 && (n->attrib_type == UFBX_ELEMENT_EMPTY || !n->attrib))
            out.points.push_back(posOf(n->node_to_world));
    }
    return true;
}

// ================================================================= the JSON samples (Blender)
bool readSceneJson(const json& j, ImportedScene& out, std::string* err) {
    if (!j.is_object() || !j.contains("frames") || !j["frames"].is_array() || j["frames"].empty()) {
        if (err) *err = j.is_object() && j.contains("error") ? j["error"].get<std::string>() : "no camera found";
        return false;
    }
    out.cameraName = j.value("camera", std::string("camera"));
    out.fps = j.value("fps", 0.0);
    out.nearClip = j.value("near", 0.1);
    out.farClip = j.value("far", 10000.0);
    for (auto& f : j["frames"]) {
        CameraSample s;
        s.frame = f.value("frame", 1.0);
        auto t = f.value("t", std::vector<double>{0, 0, 0});
        auto r = f.value("R", std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
        if (t.size() < 3 || r.size() < 9) continue;
        s.C = V3(t[0], t[1], t[2]);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) s.R.m[a][b] = r[a * 3 + b];
        s.R = cam3d::orthonormalize(s.R);
        s.focal = f.value("focal", 50.0);
        s.haperture = f.value("haperture", 24.576);
        s.vaperture = f.value("vaperture", 18.672);
        out.samples.push_back(s);
    }
    if (j.contains("points") && j["points"].is_array())
        for (auto& p : j["points"])
            if (p.is_array() && p.size() >= 3) out.points.emplace_back(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
    if (out.samples.empty()) {
        if (err) *err = "no camera samples";
        return false;
    }
    return true;
}

const char* blenderImportScript() {
    return R"PY(# Welder: a camera and its points from an Alembic / FBX file, as JSON (Y up, camera looks down -Z)
import bpy, sys, json
argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]
out = {}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if src.lower().endswith('.abc'):
        bpy.ops.wm.alembic_import(filepath=src, as_background_job=False, set_frame_range=True)
    else:
        bpy.ops.import_scene.fbx(filepath=src)
    sc = bpy.context.scene
    cams = [o for o in sc.objects if o.type == 'CAMERA']
    if not cams:
        raise RuntimeError('no camera in ' + src)
    def where(o):
        m = o.matrix_world
        return m.to_translation()
    def moved(o):
        sc.frame_set(sc.frame_start); a = where(o).copy()
        sc.frame_set(sc.frame_end); b = where(o).copy()
        return (a - b).length
    cam = max(cams, key=moved) if len(cams) > 1 else cams[0]
    def yup(v):  # Blender Z up -> Y up
        return [v[0], v[2], -v[1]]
    frames = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        m = cam.matrix_world.to_3x3().normalized()
        cols = [yup([m[0][k], m[1][k], m[2][k]]) for k in range(3)]
        R = [cols[k][i] for i in range(3) for k in range(3)]
        d = cam.data
        sw, sh = d.sensor_width, d.sensor_height
        if d.sensor_fit == 'VERTICAL':
            sw = sh * sc.render.resolution_x / max(1, sc.render.resolution_y)
        elif d.sensor_fit == 'AUTO' or sh <= 0:
            sh = sw * sc.render.resolution_y / max(1, sc.render.resolution_x)
        frames.append({'frame': f, 't': yup(where(cam)), 'R': R, 'focal': d.lens, 'haperture': sw, 'vaperture': sh})
    sc.frame_set(sc.frame_start)
    pts = []
    for o in sc.objects:
        if o == cam or o.type == 'CAMERA' or o.type == 'LIGHT':
            continue
        if o.type == 'EMPTY' and not o.children:
            pts.append(yup(o.matrix_world.to_translation()))
        elif o.type == 'MESH' and len(o.data.polygons) == 0:
            for v in o.data.vertices:
                pts.append(yup(o.matrix_world @ v.co))
    out = {'camera': cam.name, 'fps': sc.render.fps / sc.render.fps_base, 'near': cam.data.clip_start,
           'far': cam.data.clip_end, 'frames': frames, 'points': pts}
except Exception as e:
    out = {'error': str(e)}
with open(dst, 'w') as fh:
    json.dump(out, fh)
)PY";
}

bool readViaBlender(const std::string& path, ImportedScene& out, std::string* err) {
    const std::string exe = blender::findBlender("");
    if (exe.empty()) {
        if (err) *err = "reading Alembic needs Blender (install it, or set STRATA_BLENDER to blender's path)";
        return false;
    }
    namespace fs = std::filesystem;
    fs::path dir = fs::path(cacheDir("scene-import"));
    std::error_code ec;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const fs::path script = dir / ("import_" + stamp + ".py"), result = dir / ("import_" + stamp + ".json"),
                   log = dir / ("import_" + stamp + ".log");
    std::ofstream(script) << blenderImportScript();
    std::string perr;
    auto p = Process::start({exe, "-b", "--factory-startup", "--python", script.string(), "--", path, result.string()}, log.string(), &perr);
    if (!p) {
        if (err) *err = "couldn't start Blender: " + perr;
        return false;
    }
    for (int i = 0; i < 2400 && p->running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (p->running()) {
        p->kill();
        if (err) *err = "Blender took too long reading " + path;
        return false;
    }
    std::ifstream in(result);
    if (!in) {
        if (err) *err = "Blender didn't read " + path + " (see " + log.string() + ")";
        return false;
    }
    json j = json::parse(in, nullptr, false);
    in.close();
    fs::remove(script, ec);
    fs::remove(result, ec);
    fs::remove(log, ec);
    return readSceneJson(j, out, err);
}

// ================================================================= into the node
void bakeToParams(const ImportedScene& s, const std::string& file, json& p) {
    auto r6 = [](double v) { return std::round(v * 1e6) / 1e6; };
    auto curve = [&](const std::vector<std::pair<double, double>>& kv) -> json {
        bool same = true;
        for (auto& [f, v] : kv) same = same && std::fabs(v - kv[0].second) < 1e-9;
        if (same) return r6(kv[0].second);
        json keys = json::array();
        for (auto& [f, v] : kv) keys.push_back({f, r6(v)});
        return {{"keys", keys}, {"interp", "linear"}};
    };
    std::vector<std::pair<double, double>> T[3], Rr[3], F, HA, VA;
    V3 prev;
    bool have = false;
    for (auto& c : s.samples) {
        V3 e = cam3d::matrixToEuler(c.R, "ZXY", have ? &prev : nullptr);
        prev = e;
        have = true;
        for (int k = 0; k < 3; ++k) {
            T[k].push_back({c.frame, c.C[k]});
            Rr[k].push_back({c.frame, e[k]});
        }
        F.push_back({c.frame, c.focal});
        HA.push_back({c.frame, c.haperture});
        VA.push_back({c.frame, c.vaperture});
    }
    if (!have) return;
    p["translate"] = json::array({curve(T[0]), curve(T[1]), curve(T[2])});
    p["rotate"] = json::array({curve(Rr[0]), curve(Rr[1]), curve(Rr[2])});
    p["rot_order"] = "ZXY";
    p["focal"] = curve(F);
    p["haperture"] = curve(HA);
    p["vaperture"] = curve(VA);
    p["near"] = r6(s.nearClip);
    p["far"] = r6(s.farClip);
    json pts = json::array();
    for (auto& q : s.points) pts.push_back({r6(q.x), r6(q.y), r6(q.z)});
    p["points"] = pts;
    p["camera_from"] = "imported";
    p["import_file"] = file;
    char info[256];
    std::snprintf(info, sizeof info, "%s, frames %g-%g, %zu points", s.cameraName.c_str(), s.samples.front().frame,
                  s.samples.back().frame, s.points.size());
    p["import_info"] = info;
}

bool importFile(const std::string& path, double fps, double firstFrame, json& params, std::string* err) {
    (void)firstFrame;
    ImportedScene s;
    const std::string ext = lower(std::filesystem::path(path).extension().string());
    bool ok = false;
    if (ext == ".fbx") ok = readFbx(path, fps, s, err);
    else if (ext == ".abc") ok = readViaBlender(path, s, err);
    else if (err) *err = "import reads .fbx and .abc files";
    if (!ok) return false;
    bakeToParams(s, path, params);
    return true;
}

}  // namespace strata::scene3d
