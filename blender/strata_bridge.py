# Strata -> Blender render bridge. Strata's BlenderRender node runs
#
#   blender -b [scene.blend] --python-exit-code 1 -P strata_bridge.py -- job.json
#
# The job (written by Strata) holds the render settings, the Strata camera and
# geometry per frame (already in Blender's Z-up world), the passes to enable and
# the frames to render. Each frame is written as a multilayer EXR to a temporary
# name and renamed when complete, so Strata never reads a half-written file.
# Works with Blender 4.2 and later (EEVEE Next and Cycles).
import json
import math
import os
import struct
import sys
import traceback

import bpy
from mathutils import Matrix


def log(*a):
    print("STRATA:", *a, flush=True)


def load_job():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if not args:
        raise SystemExit("usage: blender -b [file.blend] -P strata_bridge.py -- job.json")
    with open(args[0], "r", encoding="utf-8") as f:
        return json.load(f)


def set_engine(scene, job):
    eng = job.get("engine", "file")
    names = {"cycles": "CYCLES", "eevee": "BLENDER_EEVEE_NEXT", "workbench": "BLENDER_WORKBENCH"}
    if eng in names:
        want = names[eng]
        try:
            scene.render.engine = want
        except TypeError:
            if want == "BLENDER_EEVEE_NEXT":  # Blender 5 renamed it back
                scene.render.engine = "BLENDER_EEVEE"
            else:
                raise
    log("engine", scene.render.engine)
    samples = int(job.get("samples", 0))
    if scene.render.engine == "CYCLES":
        cy = scene.cycles
        if samples > 0:
            cy.samples = samples
        cy.use_denoising = bool(job.get("denoise", True))
        dev = job.get("device", "auto")
        cy.device = "CPU"
        if dev in ("auto", "gpu"):
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        prefs.compute_device_type = kind
                    except TypeError:
                        continue
                    prefs.get_devices()
                    gpus = [d for d in prefs.devices if d.type == kind]
                    if gpus:
                        for d in prefs.devices:
                            d.use = d.type == kind
                        cy.device = "GPU"
                        log("device", kind, ", ".join(d.name for d in gpus))
                        break
            except Exception as e:  # no GPU support in this build
                log("no GPU:", e)
            if cy.device == "CPU":
                log("device CPU")
    elif samples > 0 and scene.render.engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = samples


def set_output(scene, job):
    r = scene.render
    r.resolution_x = int(job["width"])
    r.resolution_y = int(job["height"])
    r.resolution_percentage = 100
    pa = float(job.get("pixel_aspect", 1.0))
    r.pixel_aspect_x, r.pixel_aspect_y = (pa, 1.0) if pa >= 1 else (1.0, 1.0 / pa)
    r.film_transparent = bool(job.get("transparent", True))
    r.use_file_extension = False
    r.use_compositing = False  # the raw passes, not the .blend's compositor output
    r.use_sequencer = False
    s = r.image_settings
    s.file_format = "OPEN_EXR_MULTILAYER"
    s.color_depth = "32"
    s.exr_codec = "ZIP"
    if job.get("motion_blur") is not None:
        r.use_motion_blur = bool(job["motion_blur"])


def set_passes(scene, job):
    vl = bpy.context.view_layer
    want = set(job.get("passes", []))
    table = {
        "depth": ["use_pass_z"],
        "normal": ["use_pass_normal"],
        "vector": ["use_pass_vector"],
        "position": ["use_pass_position"],
        "mist": ["use_pass_mist"],
        "diffuse": ["use_pass_diffuse_direct", "use_pass_diffuse_indirect", "use_pass_diffuse_color"],
        "glossy": ["use_pass_glossy_direct", "use_pass_glossy_indirect", "use_pass_glossy_color"],
        "transmission": ["use_pass_transmission_direct", "use_pass_transmission_indirect", "use_pass_transmission_color"],
        "emission": ["use_pass_emit"],
        "environment": ["use_pass_environment"],
        "ao": ["use_pass_ambient_occlusion"],
        "shadow": ["use_pass_shadow_catcher"],
        "cryptomatte": ["use_pass_cryptomatte_object", "use_pass_cryptomatte_material"],
    }
    for name, props in table.items():
        on = name in want
        for p in props:
            for owner in (vl, getattr(vl, "cycles", None), getattr(vl, "eevee", None)):
                if owner is not None and hasattr(owner, p):
                    try:
                        setattr(owner, p, on)
                    except Exception:
                        pass
    if "vector" in want:
        # Cycles only writes vectors without motion blur
        scene.render.use_motion_blur = False


def key_matrix(obj, frames, bframe_of):
    obj.rotation_mode = "QUATERNION"
    prev = None
    for f, m in sorted(((int(k), v) for k, v in frames.items())):
        obj.matrix_world = Matrix(m)
        q = obj.rotation_quaternion.copy()
        if prev is not None:
            q.make_compatible(prev)
            obj.rotation_quaternion = q
        prev = q
        bf = bframe_of(f)
        obj.keyframe_insert("location", frame=bf)
        obj.keyframe_insert("rotation_quaternion", frame=bf)
        obj.keyframe_insert("scale", frame=bf)


def build_camera(scene, job, bframe_of):
    cam = job.get("camera")
    if not cam:
        if scene.camera is None:
            raise SystemExit("the .blend has no active camera and no camera is connected to the BlenderRender node")
        return
    data = bpy.data.cameras.new("StrataCamera")
    obj = bpy.data.objects.new("StrataCamera", data)
    scene.collection.objects.link(obj)
    scene.camera = obj
    data.sensor_fit = "HORIZONTAL"
    frames = cam["frames"]
    key_matrix(obj, {k: v["matrix"] for k, v in frames.items()}, bframe_of)
    for k, v in sorted(frames.items(), key=lambda kv: int(kv[0])):
        bf = bframe_of(int(k))
        data.lens = v["focal"]
        data.sensor_width = v["haperture"]
        data.sensor_height = v["vaperture"]
        data.clip_start = max(1e-4, v.get("near", 0.1))
        data.clip_end = max(data.clip_start * 10, v.get("far", 100000.0))
        data.keyframe_insert("lens", frame=bf)
        data.keyframe_insert("sensor_width", frame=bf)
    log("camera: %d keys" % len(frames))


def read_mesh(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        P = struct.unpack("<%df" % (n * 9), f.read(n * 36))
        UV = struct.unpack("<%df" % (n * 6), f.read(n * 24))
    return n, P, UV


def material_for(o, idx):
    mode = o.get("mode", "visible")
    mat = bpy.data.materials.new("Strata_%d" % idx)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if o.get("shader") == "emission":
        sh = nt.nodes.new("ShaderNodeEmission")
        col_in = sh.inputs["Color"]
    else:
        sh = nt.nodes.new("ShaderNodeBsdfPrincipled")
        col_in = sh.inputs["Base Color"]
    nt.links.new(sh.outputs[0], out.inputs["Surface"])
    tex = o.get("texture")
    if tex:
        img = bpy.data.images.load(tex["path"], check_existing=True)
        for cs in ("Linear Rec.709", "Linear"):  # Strata writes textures as linear Rec.709 EXRs
            try:
                img.colorspace_settings.name = cs
                break
            except TypeError:
                pass
        img.alpha_mode = "PREMUL"  # Strata's images are premultiplied
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = img
        proj = o.get("projection")
        if proj:
            # a camera projection (Project3D): texture coordinates from the projector
            # camera, per shading point, so it stays perspective-correct
            m = Matrix(proj["matrix"])
            C = m.translation
            g = nt.nodes.new("ShaderNodeNewGeometry")
            rel = nt.nodes.new("ShaderNodeVectorMath")
            rel.operation = "SUBTRACT"
            rel.inputs[1].default_value = C
            nt.links.new(g.outputs["Position"], rel.inputs[0])
            axis = []
            for j in range(3):
                d = nt.nodes.new("ShaderNodeVectorMath")
                d.operation = "DOT_PRODUCT"
                d.inputs[1].default_value = (m[0][j], m[1][j], m[2][j])
                nt.links.new(rel.outputs["Vector"], d.inputs[0])
                axis.append(d.outputs["Value"])

            def math(op, a, b):
                n = nt.nodes.new("ShaderNodeMath")
                n.operation = op
                for k, v in enumerate((a, b)):
                    if isinstance(v, (int, float)):
                        n.inputs[k].default_value = v
                    else:
                        nt.links.new(v, n.inputs[k])
                return n.outputs["Value"]

            u = math("ADD", math("MULTIPLY", math("DIVIDE", axis[0], axis[2]), -proj["fx"]), 0.5)
            v = math("ADD", math("MULTIPLY", math("DIVIDE", axis[1], axis[2]), -proj["fy"]), 0.5)
            comb = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(u, comb.inputs[0])
            nt.links.new(v, comb.inputs[1])
            nt.links.new(comb.outputs["Vector"], tn.inputs["Vector"])
            tn.extension = "CLIP"
        if tex.get("sequence"):
            img.source = "SEQUENCE"
            tn.image_user.frame_duration = int(tex["count"])
            tn.image_user.frame_start = int(tex["start"])
            tn.image_user.frame_offset = int(tex["offset"])
            tn.image_user.use_auto_refresh = True
        nt.links.new(tn.outputs["Color"], col_in)
        if "Alpha" in sh.inputs:
            nt.links.new(tn.outputs["Alpha"], sh.inputs["Alpha"])
    return mat


def build_objects(scene, job, bframe_of):
    for i, o in enumerate(job.get("objects", [])):
        n, P, UV = read_mesh(o["mesh"])
        me = bpy.data.meshes.new(o.get("name", "strata"))
        me.vertices.add(n * 3)
        me.vertices.foreach_set("co", P)
        me.loops.add(n * 3)
        me.loops.foreach_set("vertex_index", list(range(n * 3)))
        me.polygons.add(n)
        me.polygons.foreach_set("loop_start", list(range(0, n * 3, 3)))
        try:  # Blender < 4.0 needs the sizes too; later versions derive them
            me.polygons.foreach_set("loop_total", [3] * n)
        except Exception:
            pass
        me.update(calc_edges=True)
        uv = me.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", UV)
        me.validate()
        obj = bpy.data.objects.new(o.get("name", "strata"), me)
        scene.collection.objects.link(obj)
        obj.data.materials.append(material_for(o, i))
        mode = o.get("mode", "visible")
        if mode == "shadow catcher":
            obj.is_shadow_catcher = True
        elif mode == "holdout":
            obj.is_holdout = True
        key_matrix(obj, o["frames"], bframe_of)
    if job.get("objects"):
        log("objects: %d" % len(job["objects"]))


def main():
    job = load_job()
    if job.get("blend"):
        log("scene", job["blend"])
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        world = bpy.data.worlds.new("World")
        scene.world = world
        world.color = (0.05, 0.05, 0.05)
        if world.node_tree is None:
            world.use_nodes = True
    scene = bpy.context.scene
    offset = int(job.get("frame_offset", 0))

    def bframe_of(f):
        return f + offset

    set_engine(scene, job)
    set_output(scene, job)
    set_passes(scene, job)
    build_camera(scene, job, bframe_of)
    build_objects(scene, job, bframe_of)
    if job.get("python"):
        log("running the node's python")
        exec(compile(job["python"], "BlenderRender python", "exec"), {"bpy": bpy, "scene": scene, "job": job})
    for item in job["frames"]:
        out = item["out"]
        tmp = out + ".part.exr"
        scene.frame_set(bframe_of(int(item["frame"])))
        scene.render.filepath = tmp
        log("rendering frame %d (Blender frame %d)" % (item["frame"], scene.frame_current))
        bpy.ops.render.render(write_still=True)
        if not os.path.exists(tmp):
            raise SystemExit("Blender wrote no image for frame %d" % item["frame"])
        os.replace(tmp, out)
        log("done frame %d" % item["frame"])


try:
    main()
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)
