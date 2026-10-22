"""Tests for tools/nk2strata.py.  Run: python3 -m unittest tests/test_nk2strata.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nk2strata  # noqa: E402


def convert(text):
    with tempfile.NamedTemporaryFile("w", suffix=".nk", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        comp, report = nk2strata.convert(path)
    finally:
        os.unlink(path)
    nodes = {n["name"]: n for n in comp["nodes"]}
    return comp, report, nodes


class Tokenizer(unittest.TestCase):
    def test_braces_quotes_brackets(self):
        cmds = nk2strata.split_commands('Grade {\n white {1 2 {3}}\n label "a \\"b\\" c"\n}\nset N1 [stack 0]\n')
        self.assertEqual(str(cmds[0][0]), "Grade")
        self.assertEqual(cmds[0][1].kind, "brace")
        self.assertEqual([str(w) for w in cmds[1]], ["set", "N1", "[stack 0]"])
        knobs = nk2strata.split_commands(str(cmds[0][1]))
        self.assertEqual(str(knobs[1][1]), 'a "b" c')

    def test_curves(self):
        w = nk2strata.Word("curve x1 0 x10 1", "brace")
        self.assertEqual(nk2strata.scalar(w), {"keys": [[1.0, 0.0], [10.0, 1.0]], "interp": "smooth"})
        w = nk2strata.Word("curve L x5 2 3 4", "brace")
        self.assertEqual(nk2strata.scalar(w), {"keys": [[5.0, 2.0], [6.0, 3.0], [7.0, 4.0]], "interp": "linear"})
        w = nk2strata.Word("curve K x1 0 s0 x24 1 s0", "brace")
        self.assertEqual(nk2strata.scalar(w)["keys"], [[1.0, 0.0], [24.0, 1.0]])
        self.assertEqual(nk2strata.scalar(nk2strata.Word("curve x3 7", "brace")), 7.0)

    def test_vectors_and_expressions(self):
        v = nk2strata.vector(nk2strata.Word("{curve x1 0 x2 10} 5", "brace"))
        self.assertEqual(v[1], 5.0)
        self.assertEqual(v[0]["keys"][1], [2.0, 10.0])
        with self.assertRaises(nk2strata.Expression):
            nk2strata.vector(nk2strata.Word("{parent.size}", "brace"))


class Graph(unittest.TestCase):
    def test_stack_order_top_is_input0(self):
        _, _, n = convert("""
Constant { inputs 0 name A_side }
Constant { inputs 0 name B_side }
Merge2 { inputs 2 name M }
""")
        self.assertEqual(n["M"]["inputs"], ["B_side", "A_side"])  # input 0 = B = top of stack

    def test_set_push_and_mask(self):
        _, _, n = convert("""
Constant { inputs 0 name Src }
set N1 [stack 0]
Radial { inputs 0 name Matte }
push $N1
Grade { inputs 1+1 white 2 name G }
Constant { inputs 0 name Over }
push $N1
Merge2 { inputs 2 name M }
""")
        self.assertEqual(n["G"]["inputs"], ["Src", "Matte"])
        self.assertEqual(n["M"]["inputs"], ["Src", "Over"])

    def test_merge_mask_slot(self):
        _, _, n = convert("""
Radial { inputs 0 name K }
Constant { inputs 0 name A }
Constant { inputs 0 name B }
Merge2 { inputs 2+1 operation plus maskChannelInput rgba.alpha name M }
""")
        self.assertEqual(n["M"]["inputs"], ["B", "A", "K"])
        self.assertEqual(n["M"]["params"]["operation"], "plus")

    def test_push_zero_leaves_input_empty(self):
        _, _, n = convert("""
push 0
Constant { inputs 0 name B }
Merge2 { inputs 2 name M }
""")
        self.assertEqual(n["M"]["inputs"], ["B"])

    def test_group_stays_a_group(self):
        _, _, n = convert("""
Constant { inputs 0 name Src }
Group { name G1 xpos 10 ypos 20 }
 Input { inputs 0 name Input1 }
 Grade { white 2 name Inner }
 Output { name Output1 }
end_group
Blur { size 3 name After }
""")
        self.assertEqual(n["G1"]["type"], "Group")
        self.assertEqual(n["G1"]["inputs"], ["Src"])
        self.assertEqual(n["Input1"]["group"], "G1")
        self.assertEqual(n["Input1"]["params"]["number"], 0)
        self.assertEqual(n["Inner"]["inputs"], ["Input1"])
        self.assertEqual(n["Inner"]["group"], "G1")
        self.assertEqual(n["Output1"]["inputs"], ["Inner"])
        self.assertEqual(n["After"]["inputs"], ["G1"])
        self.assertNotIn("group", n["After"])

    def test_studio_slate_generator_becomes_a_slate(self):
        # the studio names its slate gizmo in studio.json
        cfg = os.path.join(tempfile.mkdtemp(), "studio.json")
        with open(cfg, "w") as f:
            json.dump({"nk": {"slate_gizmos": ["ACME_SLATE_GENERATOR"]}}, f)
        old = os.environ.get("STRATA_STUDIO_CONFIG")
        os.environ["STRATA_STUDIO_CONFIG"] = cfg
        try:
            # as a gizmo
            _, _, n = convert("""
Constant { inputs 0 name Plate }
ACME_SLATE_GENERATOR {
 Notes_Input "Single frame warp fix"
 Mattes_Input N/A
 _Image_Location {-1450 64}
 _Image_scale 0.37
 SlGen_Vendor_Name ACME
 SlGen_Show_Name "Demo Show"
 Socket_Artist_Name "A. Artist"
 SlGen_Version "[join [lrange [split [file rootname [file tail [value root.name]]] _ ] 5 5] _ ]"
 name ACME_SLATE_GENERATOR
}
Write { file out.####.exr name W }
""")
            s = n["ACME_SLATE_GENERATOR"]
            self.assertEqual(s["type"], "Slate")
            self.assertEqual(s["inputs"], ["Plate"])
            self.assertEqual(s["params"]["notes"], "Single frame warp fix")
            self.assertEqual(s["params"]["show"], "Demo Show")
            self.assertEqual(s["params"]["artist"], "A. Artist")
            self.assertEqual(s["params"]["image_location"], [-1450.0, 64.0])
            self.assertAlmostEqual(s["params"]["image_scale"], 0.37)
            self.assertNotIn("version", s["params"])  # the expression: worked out by the Slate
            self.assertEqual(n["W"]["inputs"], ["ACME_SLATE_GENERATOR"])
            # as a Group of that name: its insides aren't brought in
            _, _, n = convert("""
Constant { inputs 0 name Plate }
Group { name ACME_SLATE_GENERATOR1 Notes_Input "hello" }
 Input { inputs 0 name Input1 }
 Text { message "filename :" name Line_One_TAG }
 Output { name Output1 }
end_group
Blur { size 3 name After }
""")
            self.assertEqual(n["ACME_SLATE_GENERATOR1"]["type"], "Slate")
            self.assertEqual(n["ACME_SLATE_GENERATOR1"]["params"]["notes"], "hello")
            self.assertNotIn("Line_One_TAG", n)
            self.assertEqual(n["After"]["inputs"], ["ACME_SLATE_GENERATOR1"])
        finally:
            if old is None:
                os.environ.pop("STRATA_STUDIO_CONFIG", None)
            else:
                os.environ["STRATA_STUDIO_CONFIG"] = old

    def test_nested_groups_and_name_clashes(self):
        _, _, n = convert("""
Constant { inputs 0 name C }
Blur { size 2 name Blur1 }
Group { name Outer }
 Input { inputs 0 name Input1 }
 Blur { size {{Blur1.size*2}} name Blur1 }
 Group { name Inner }
  Input { inputs 0 name Input1 }
  Grade { multiply {{parent.parent.amount}} name G }
  Output { name Output1 }
 end_group
 Output { name Output1 }
end_group
""")
        self.assertEqual(n["Blur1_2"]["group"], "Outer")   # the inner Blur1, renamed
        self.assertEqual(n["Blur1_2"]["params"]["size"], {"expr": "Blur1_2.size*2"})  # its own scope first
        self.assertEqual(n["Inner"]["group"], "Outer")
        self.assertEqual(n["G"]["group"], "Inner")
        self.assertEqual(n["G"]["params"]["multiply"], {"expr": "parent.parent.amount"})

    def test_backdrop_and_sticky_note(self):
        comp, _, n = convert("""
BackdropNode { inputs 0 name BD tile_color 0x7171c600 label "<b>plates</b>" note_font_size 42 xpos 1 ypos 2 bdwidth 400 bdheight 250 z_order 1 }
StickyNote { inputs 0 name Note label "fix the edge" }
Constant { inputs 0 name C }
Viewer { name Viewer1 }
""")
        self.assertEqual(sorted(n), ["BD", "C", "Note"])
        bd = n["BD"]
        self.assertEqual(bd["type"], "Backdrop")
        self.assertEqual(bd["params"]["label"], "plates")
        self.assertEqual(bd["params"]["note_font_size"], 42)
        self.assertAlmostEqual(bd["params"]["color"][2], 0xc6 / 255.0)
        self.assertEqual((bd["ui"]["w"], bd["ui"]["h"]), (400, 250))
        self.assertEqual(n["Note"]["type"], "StickyNote")
        self.assertEqual(n["Note"]["params"]["label"], "fix the edge")


class Knobs(unittest.TestCase):
    def test_root(self):
        comp, _, _ = convert('Root { inputs 0 first_frame 1001 last_frame 1100 fps 23.976 '
                             'format "2048 1152 0 0 2048 1152 1 MyFmt" }\n')
        self.assertEqual(comp["format"]["width"], 2048)
        self.assertEqual(comp["frame_range"], [1001.0, 1100.0])
        self.assertAlmostEqual(comp["fps"], 23.976)

    def test_grade_names(self):
        _, _, n = convert("Constant { inputs 0 name C }\nGrade { black 0.1 white {1 2 3 1} add 0.5 gamma 2 name G }\n")
        p = n["G"]["params"]
        self.assertEqual(p["lift"], 0.1)
        self.assertEqual(p["gain"], [1.0, 2.0, 3.0, 1.0])
        self.assertEqual(p["offset"], 0.5)

    def test_transform_animation(self):
        _, _, n = convert("Constant { inputs 0 name C }\n"
                          "Transform { translate {{curve x1 0 x10 100} 5} rotate 30 scale 2 name T }\n")
        p = n["T"]["params"]
        self.assertEqual(p["translate"][0]["keys"], [[1.0, 0.0], [10.0, 100.0]])
        self.assertEqual(p["translate"][1], 5.0)
        self.assertEqual(p["scale"], 2.0)

    def test_expressions_are_translated(self):
        _, report, n = convert("Constant { inputs 0 name C }\nBlur { size {{C.color.r*10}} name B }\n"
                               "Grade { white {{B.size/10}} multiply {{frame} {[value B.size]} 1 1} name G }\n"
                               "Transform { translate {{G.white} {G.white}} rotate {{curve x1 0 x10 90}} name T }\n")
        self.assertEqual(n["B"]["params"]["size"], {"expr": "C.color.r*10"})
        self.assertEqual(n["G"]["params"]["gain"], {"expr": "B.size/10"})
        self.assertEqual(n["G"]["params"]["multiply"], [{"expr": "frame"}, {"expr": "[value B.size]"}, 1.0, 1.0])
        # a per-component link reads the same component, under Welder's knob name (white -> gain)
        self.assertEqual(n["T"]["params"]["translate"], [{"expr": "G.gain.0"}, {"expr": "G.gain.1"}])
        self.assertEqual(n["T"]["params"]["rotate"]["keys"], [[1.0, 0.0], [10.0, 90.0]])
        self.assertFalse(any("expression" in w["message"] for w in report["warnings"]))

    def test_tcl_expression_warns_and_keeps_default(self):
        _, report, n = convert('Constant { inputs 0 name C }\nBlur { size {{"[python thisNode().width()]"}} name B }\n')
        self.assertNotIn("size", n["B"]["params"])
        self.assertTrue(any("expression" in w["message"] for w in report["warnings"]))

    def test_group_user_knobs_and_links(self):
        _, report, n = convert("""
Constant { inputs 0 name C }
Group {
 name Look
 addUserKnob {20 User}
 addUserKnob {7 amount l Amount R 0 10}
 amount 4
 addUserKnob {4 mode M {soft hard}}
 mode hard
 addUserKnob {6 on +STARTLINE}
 on true
 addUserKnob {41 size l Softness T Soft.size}
 addUserKnob {41 tint T Tint.white}
}
 Input { inputs 0 name Input1 }
 Blur { size 7 name Soft }
 Grade { white {1 0.5 0.25 1} name Tint }
 Grade { multiply {{parent.amount}} name Amt }
 Output { name Output1 }
end_group
""")
        g = n["Look"]["params"]
        names = [k["name"] for k in g["knobs"]]
        self.assertEqual(names, ["amount", "mode", "on", "size", "tint"])
        self.assertEqual(g["amount"], 4.0)
        self.assertEqual(g["mode"], "hard")
        self.assertIs(g["on"], True)
        k = {d["name"]: d for d in g["knobs"]}
        self.assertEqual((k["amount"]["lo"], k["amount"]["hi"], k["amount"]["label"]), (0, 10, "Amount"))
        self.assertEqual(k["mode"]["options"], ["soft", "hard"])
        # link knobs: the group holds the value, the inner knob reads it
        self.assertEqual(g["size"], 7.0)
        self.assertEqual(n["Soft"]["params"]["size"], {"expr": "parent.size"})
        self.assertEqual(g["tint"], [1.0, 0.5, 0.25, 1.0])
        self.assertEqual(k["tint"]["type"], "color")
        self.assertEqual(n["Tint"]["params"]["gain"][2], {"expr": "parent.tint.2"})
        self.assertEqual(n["Amt"]["params"]["multiply"], {"expr": "parent.amount"})

    def test_gizmo_from_the_gizmo_path(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "gizmos"))
        with open(os.path.join(d, "gizmos", "SoftGlow.gizmo"), "w") as f:
            f.write("""version 13.2 v1
Gizmo {
 inputs 1
 addUserKnob {20 User}
 addUserKnob {7 glow R 0 100}
 glow 10
}
 Input { inputs 0 name Input1 }
 Blur { size {{parent.glow}} name Blur1 }
 Merge2 { inputs 2 operation plus name Merge1 }
 Output { name Output1 }
end_group
""")
        old = os.environ.get("STRATA_GIZMO_PATH")
        os.environ["STRATA_GIZMO_PATH"] = d
        try:
            _, report, n = convert("Constant { inputs 0 name C }\nSoftGlow { glow 25 name SoftGlow1 }\nBlur { name After }\n")
        finally:
            if old is None:
                del os.environ["STRATA_GIZMO_PATH"]
            else:
                os.environ["STRATA_GIZMO_PATH"] = old
        g = n["SoftGlow1"]
        self.assertEqual(g["type"], "Group")
        self.assertEqual(g["params"]["gizmo"], "SoftGlow.gizmo")
        self.assertEqual(g["params"]["glow"], 25.0)   # the instance's value over the gizmo's default
        self.assertEqual(g["inputs"], ["C"])
        self.assertEqual(n["Blur1"]["group"], "SoftGlow1")
        self.assertEqual(n["Blur1"]["params"]["size"], {"expr": "parent.glow"})
        self.assertEqual(n["After"]["inputs"], ["SoftGlow1"])

    def test_clipboard_fragment(self):
        clip = """set cut_paste_input [stack 0]
version 14.0 v5
push $cut_paste_input
Blur {
 size 12
 name Blur7
 selected true
 xpos 100
 ypos -50
}
Grade {
 white {{Blur7.size/12}}
 name Grade3
 selected true
 xpos 100
 ypos 0
}
"""
        with tempfile.NamedTemporaryFile("w", suffix=".nk", delete=False) as f:
            f.write(clip)
            src = f.name
        out = src + ".json"
        try:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "nk2strata.py"), src, "--fragment", "-o", out, "-q"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            j = json.load(open(out))
        finally:
            os.unlink(src)
            if os.path.exists(out):
                os.unlink(out)
        nodes = {x["name"]: x for x in j["strata_nodes"]}
        self.assertEqual(nodes["Blur7"]["inputs"], [])
        self.assertEqual(nodes["Grade3"]["inputs"], ["Blur7"])
        self.assertEqual(nodes["Grade3"]["params"]["gain"], {"expr": "Blur7.size/12"})
        self.assertEqual(nodes["Blur7"]["ui"]["x"], 100)

    def test_shuffle_old(self):
        _, _, n = convert("Constant { inputs 0 name C }\nShuffle { red blue blue red alpha white name S }\n")
        self.assertEqual(n["S"]["params"]["mapping"], [["R", "B"], ["B", "R"], ["A", "1"]])

    def test_ofx_node(self):
        _, _, n = convert("Constant { inputs 0 name C }\n"
                          "OFXcom.genarts.sapphire.blur_v1 { amount 0.5 mode Fast name Sapph }\n")
        p = n["Sapph"]["params"]
        self.assertEqual(n["Sapph"]["type"], "OFX")
        self.assertEqual(p["plugin"], "com.genarts.sapphire.blur")
        self.assertEqual(p["version"], 1)
        self.assertEqual(p["values"], {"amount": 0.5, "mode": "Fast"})

    def test_disabled_node_stays_disabled(self):
        _, _, n = convert("Constant { inputs 0 name C }\nBlur { size 9 disable true name B }\n")
        self.assertEqual(n["B"]["type"], "Blur")
        self.assertTrue(n["B"]["params"]["disable"])

    def test_unsupported_is_reported(self):
        _, report, n = convert("Constant { inputs 0 name C }\nPrimatte3 { name K }\n")
        self.assertEqual(n["K"]["type"], "Unsupported:Primatte3")
        self.assertEqual(report["unsupported_classes"], {"Primatte3": 1})


class EndToEnd(unittest.TestCase):
    """The example script translates and loads in the Welder CLI (if built)."""

    def test_example_loads(self):
        exe = os.path.join(ROOT, "build", "strata")
        nk = os.path.join(ROOT, "examples", "nk", "shot010_comp_v003.nk")
        comp, report = nk2strata.convert(nk)
        self.assertEqual(report["nodes_total"], 23)  # with the backdrop and the group's Input / Output
        if not os.path.exists(exe):
            self.skipTest("strata not built")
        with tempfile.NamedTemporaryFile("w", suffix=".weld", delete=False, dir=os.path.dirname(nk)) as f:
            json.dump(comp, f)
            path = f.name
        try:
            out = subprocess.run([exe, "info", path], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("Merge_Glow", out.stdout)
        finally:
            os.unlink(path)



ROTO_NK = r"""Roto {
 inputs 0
 output rgba
 curves {{{v x3f99999a}
  {f 0}
  {n
   {layer Root
    {f 2097152}
    {t x44700000 x44070000}
    {a pt1x 0 pt1y 0}
    {curvegroup Bezier1 512 bezier
     {{cc
       {f 8192}
       {px x447a0000
        {0 0}
        {x42c80000 x42c80000}
        {x41200000 0}
        {0 0}
        {x43480000 x42c80000}
        {0 0}
        {0 0}
        {x43480000 x43480000}
        {0 0}
        {0 0}
        {x42c80000 x43480000}
        {0 0}}}
      {cc
       {f 8192}
       {px x447a0000
        {0 0}
        {xc1200000 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}
        {0 0}}}}
     {tx x447a0000 x43160000 x43160000 0 x41a00000 0}
     {a osw x41200000 osf 0 str 1 spx x44700000 spy x44070000 sb 1 tt x40800000 opc x3f000000 ltt 2 ltn x447a0000}}
    {curvegroup Bezier2 512 bezier
     {{cc
       {f 8192}
       {px {x447a0000 x447c0000}
        {0 0}
        {{0 x41200000} {0 x41200000}}
        {0 0}
        {0 0}
        {{x41a00000 x41f00000} 0}
        {0 0}
        {0 0}
        {{x41a00000 x41f00000} x41a00000}
        {0 0}}}
      idem}
     {tx x447a0000 0 0}
     {a tt x40800000}}
    {curvegroup Brush1 1051136 bspline
     {{cc
       {f 2080}
       {px x447a0000
        {x42200000 x42200000 1}
        {x42a00000 x42200000 x3f000000}}}
      idem}
     {tx x447a0000 0 0}
     {a bs x41f00000 h x3e4ccccd r 1 g 0 b 0 a 1 tt x40e00000}}}}}}
 name Roto1
}
"""


class RotoImport(unittest.TestCase):
    def test_roto_curves(self):
        _, report, n = convert(ROTO_NK)
        r = n["Roto1"]
        self.assertEqual(r["type"], "Roto")
        p = r["params"]
        self.assertEqual(p["output"], "rgba")
        shapes = p["shapes"]
        self.assertEqual([s["name"] for s in shapes], ["Brush1", "Bezier2", "Bezier1"])
        b1 = shapes[2]
        self.assertEqual(b1["points"][0][:2], [100.0, 100.0])
        self.assertEqual(b1["points"][0][4:6], [10.0, 0.0])          # right tangent
        self.assertEqual(b1["points"][0][6:8], [-10.0, 0.0])         # feather offset
        self.assertEqual(b1["points"][2][:2], [200.0, 200.0])
        self.assertEqual(b1["opacity"], 0.5)
        self.assertEqual(b1["life"]["type"], "single")
        self.assertEqual(b1["life"]["start"], 1000.0)
        self.assertEqual(b1["transform"]["translate"], [20.0, 0])
        self.assertEqual(b1["transform"]["center"], [150.0, 150.0])
        b2 = shapes[1]
        self.assertEqual([k[0] for k in b2["anim"]], [1000.0, 1008.0])
        self.assertEqual(b2["anim"][0][1][1][0], 20.0)
        self.assertEqual(b2["anim"][1][1][1][0], 30.0)
        self.assertEqual(b2["anim"][1][1][0][1], 10.0)
        brush = shapes[0]
        self.assertEqual(brush["kind"], "stroke")
        self.assertEqual(brush["tool"], "paint")
        self.assertEqual(brush["points"], [[40.0, 40.0, 1.0], [80.0, 40.0, 0.5]])
        self.assertEqual(brush["size"], 30.0)
        self.assertEqual(brush["color"], [1.0, 0.0, 0.0, 1.0])
        self.assertTrue(any("roto" in w["message"] for w in report["warnings"]))

    def test_roto_layers_and_matrix(self):
        # Root (only a pivot: dropped) > L1 (translate 10, extra matrix: translate 5, 7) > L2 (translate 3) > Bezier1
        ident = " ".join(["0"] * 10)
        m_rows = "1 0 0 5 0 1 0 7 0 0 1 0 0 0 0 1"
        text = ("Roto {\n inputs 0\n curves {{{v x3f99999a}\n {f 0}\n {n\n"
                "  {layer Root\n   {f 2097152}\n   {t 100 50}\n"
                "   {layer L1\n    {f 0}\n    {t 0 0 0 10 0 0 1 1 1 0 0 0 0.1 0 0 2 0 0 0 " + m_rows + "}\n"
                "    {layer L2\n     {f 0}\n     {t 0 0 0 3 0 0 1 1 1 0 0 0}\n     {a vis 0}\n"
                "     {curvegroup Bezier1 512 bezier\n      {{cc\n        {f 8192}\n        {px x447a0000\n"
                "         {0 0} {1 2} {0 0} {0 0} {3 4} {0 0} {0 0} {5 6} {0 0}}}\n       idem}\n"
                "      {tx x447a0000 0 0}\n      {a opc 1}}}}}}}}\n name RotoL\n}\n")
        _, report, n = convert(text)
        items = n["RotoL"]["params"]["shapes"]
        by = {i["name"]: i for i in items}
        self.assertNotIn("Root", by)  # a layer that moves nothing isn't kept
        self.assertEqual(by["L1"]["kind"], "layer")
        self.assertNotIn("parent", by["L1"])
        self.assertEqual(by["L1"]["transform"]["translate"], [10.0, 0.0])
        self.assertEqual(by["L1"]["transform"]["skew"], [0.1, 0.0])
        self.assertEqual(by["L1"]["transform"]["extra_translate"], [2.0, 0.0])
        self.assertEqual(by["L1"]["matrix"], [1.0, 0.0, 5.0, 0.0, 1.0, 7.0, 0.0, 0.0, 1.0])
        self.assertEqual(by["L2"]["parent"], "L1")
        self.assertIs(by["L2"]["visible"], False)
        self.assertEqual(by["Bezier1"]["parent"], "L2")
        # layers come before what's in them
        names = [i["name"] for i in items]
        self.assertLess(names.index("L1"), names.index("L2"))
        self.assertLess(names.index("L2"), names.index("Bezier1"))
        self.assertFalse(any("not applied" in w["message"] for w in report["warnings"]))

    def test_roto_matrix_written_either_way(self):
        # a corner-pin-like matrix with perspective, written column-major (translation at 12, 13)
        col = [1.2, 0.1, 0, 0.0005, 0.05, 0.9, 0, 0.0002, 0, 0, 1, 0, 40, -30, 0, 1]
        mats = {}

        def chan(i, d):
            return col[i - 19] if 19 <= i < 35 else d
        mats["col"] = nk2strata.roto_matrix(chan)
        self.assertEqual(mats["col"], [1.2, 0.05, 40, 0.1, 0.9, -30, 0.0005, 0.0002, 1])
        rowm = [col[c * 4 + r] for r in range(4) for c in range(4)]  # the same matrix, row-major

        def chan2(i, d):
            return rowm[i - 19] if 19 <= i < 35 else d
        self.assertEqual(nk2strata.roto_matrix(chan2), mats["col"])
        self.assertIsNone(nk2strata.roto_matrix(lambda i, d: d))

    def test_hex_floats(self):
        self.assertEqual(nk2strata.roto_float("x3f800000"), 1.0)
        self.assertEqual(nk2strata.roto_float("xc1200000"), -10.0)
        self.assertEqual(nk2strata.roto_float("0.25"), 0.25)


REFLINK_NK = r"""CornerPin2D {
 inputs 0
 to1 {{curve x1001 500 520} {curve x1001 300 310}}
 to2 {{curve x1001 900 920} {curve x1001 300 310}}
 to3 {{curve x1001 900 920} {curve x1001 700 710}}
 to4 {{curve x1001 500 520} {curve x1001 700 710}}
 name Mocha_Pin
}
CornerPin2D {
 to1 {{input.to1} {input.to1}}
 to2 {{input.to2} {input.to2}}
 to3 {{input.to3} {input.to3}}
 to4 {{input.to4} {input.to4}}
 name INPUT_PIN_0
}
CheckerBoard2 {
 inputs 0
 name Plate
}
CornerPin2D {
 to1 {{INPUT_PIN_0.to1} {INPUT_PIN_0.to1}}
 to2 {{INPUT_PIN_0.to2} {INPUT_PIN_0.to2}}
 to3 {{INPUT_PIN_0.to3} {INPUT_PIN_0.to3}}
 to4 {{INPUT_PIN_0.to4} {INPUT_PIN_0.to4}}
 invert false
 motionblur 1
 shutteroffset centred
 from1 {{to1(ref)} {to1(ref)}}
 from2 {{to2(ref)} {to2(ref)}}
 from3 {{to3(ref)} {to3(ref)}}
 from4 {{to4(ref)} {to4(ref)}}
 name LINKED_PIN_1
 addUserKnob {20 Branch}
 addUserKnob {3 ref l "reference frame"}
 ref 1002
}
Transform {
 inputs 0
 translate {{parent.INPUT_TRANSFORM_1.translate-parent.INPUT_TRANSFORM_1.translate(ref)} {parent.INPUT_TRANSFORM_1.translate-parent.INPUT_TRANSFORM_1.translate(ref)}}
 rotate {{parent.INPUT_TRANSFORM_1.rotate-parent.INPUT_TRANSFORM_1.rotate(ref)}}
 name LINKED_TRANSFORM_1
 addUserKnob {20 Branch}
 addUserKnob {3 ref l "reference frame"}
 ref 1010
}
Transform {
 inputs 0
 translate {{input.translate.x} {input.translate.y}}
 name INPUT_TRANSFORM_1
}
Glow_Exponential {
 inputs 1
 size 25
 falloff 0.9
 falloff_color {1 0.4 0.1}
 glow_only true
 crop false
 name Glow_Exponential1
}
"""

WELDER_NK = r"""set cut_paste_input [stack 0]
push $cut_paste_input
Group {
name welder1
addUserKnob {20 giftWrap l welder}
addUserKnob {4 influence M {flat keyed}}
influence keyed
addUserKnob {41 amount l expGlow T expGlow4.amount}
addUserKnob {4 op M {over plus}}
op plus
addUserKnob {4 bboxc l bbox -STARTLINE M {A B union ""}}
bboxc union
addUserKnob {4 view l output M {output keyed key+blur "wrap only" ""}}
}
Input {
 inputs 0
 name A
}
Keyer {
 operation "luminance key"
 range {0.01 0.5 1 1}
 name mainKeyer
}
Group {
 name expGlow4
 addUserKnob {7 amount R 0 60}
 amount 22
}
 Input {
  inputs 0
  name Input1
 }
 Output {
  name Output1
 }
end_group
Blur {
 size 80
 mix 0.2
 name mainBlur
}
Merge2 {
 mix 0.3
 name mainMix
}
Output {
 name Output1
}
end_group
"""


class GizmoNativeImport(unittest.TestCase):
    def test_reference_links(self):
        _, report, n = convert(REFLINK_NK)
        hub = n["INPUT_PIN_0"]
        self.assertEqual(hub["type"], "ReferenceLink")
        self.assertEqual(hub["inputs"], [None, "Mocha_Pin"])  # a hub: follows what's on its input
        lp = n["LINKED_PIN_1"]
        self.assertEqual(lp["type"], "ReferenceLink")
        self.assertEqual(lp["inputs"], ["Plate", "INPUT_PIN_0"])  # linked to the hub (written after it)
        self.assertEqual(lp["params"]["reference_frame"], 1002)
        self.assertEqual(lp["params"]["motionblur"], 1)
        self.assertEqual(lp["params"]["shutteroffset"], "centred")
        lt = n["LINKED_TRANSFORM_1"]
        self.assertEqual(lt["type"], "ReferenceLink")
        self.assertEqual(lt["inputs"][1], "INPUT_TRANSFORM_1")  # linked to a node further down the script
        self.assertEqual(lt["params"]["reference_frame"], 1010)
        self.assertEqual(n["INPUT_TRANSFORM_1"]["type"], "ReferenceLink")
        self.assertEqual(n["Mocha_Pin"]["type"], "CornerPin2D")  # an ordinary corner pin stays one
        g = n["Glow_Exponential1"]
        self.assertEqual(g["type"], "ExpoGlow")
        self.assertEqual(g["params"]["size"], 25)
        self.assertEqual(g["params"]["falloff_color"], [1, 0.4, 0.1])
        self.assertIs(g["params"]["glow_only"], True)
        self.assertIs(g["params"]["crop"], False)

    def test_welder_group(self):
        _, report, n = convert(WELDER_NK)
        self.assertEqual(list(n), ["welder1"])  # its insides aren't needed
        w = n["welder1"]
        self.assertEqual(w["type"], "Welder")
        p = w["params"]
        self.assertEqual(p["influence"], "keyed")
        self.assertEqual(p["operation"], "plus")
        self.assertEqual(p["bbox"], "union")
        self.assertEqual(p["amount"], 22)
        self.assertEqual(p["size"], 80)
        self.assertEqual(p["extra_mix"], 0.2)
        self.assertEqual(p["mix"], 0.3)
        self.assertEqual(p["range"], [0.01, 0.5, 1, 1])


class KeyerImport(unittest.TestCase):
    def test_keylight_ibk_keyer(self):
        text = ("Constant { inputs 0 color {0.1 0.6 0.15 1} name Plate }\n"
                "OFXuk.co.thefoundry.keylight.keylight_v201 { show \"Final Result\" screenColour {0.1 0.6 0.15} "
                "screenGain 1.1 screenClipMax 0.9 name Keylight1 }\n"
                "push $cut\n"
                "Keyer { operation \"greenscreen keyer\" range {0.1 0.2 1 1} name Keyer1 }\n")
        text = text.replace("push $cut\n", "")
        _, _, n = convert(text)
        kl = n["Keylight1"]
        self.assertEqual(kl["type"], "ScreenKeyer")
        self.assertEqual(kl["params"]["screen_color"], [0.1, 0.6, 0.15, 1.0])
        self.assertEqual(kl["params"]["screen_gain"], 1.1)
        self.assertEqual(kl["params"]["clip_white"], 0.9)
        self.assertEqual(kl["params"]["output"], "final")
        ky = n["Keyer1"]
        self.assertEqual(ky["params"]["operation"], "greenscreen keyer")
        self.assertEqual(ky["params"]["range"], [0.1, 0.2, 1, 1])
        _, _, n2 = convert("Constant { inputs 0 name P }\nIBKColourV3 { screen_type blue erode 3 name C }\n"
                           "IBKGimmeV3 { inputs 2 screen_type blue name G }\n")
        self.assertEqual(n2["C"]["type"], "IBKColour")
        self.assertEqual(n2["C"]["params"]["screen_type"], "blue")
        self.assertEqual(n2["G"]["type"], "IBKGimme")


TRACKER_NK = """Tracker4 {
 inputs 0
 tracks { { 1 31 2 }
{ { 5 1 20 enable e 1 }
{ 3 1 75 name name 1 }
{ 2 1 58 track_x track_x 1 }
{ 2 1 58 track_y track_y 1 }
{ 2 1 63 offset_x offset_x 1 }
{ 2 1 63 offset_y offset_y 1 }
{ 4 1 27 T T 1 }
{ 4 1 27 R R 1 }
{ 4 1 27 S S 1 }
{ 2 0 45 error error 1 }
{ 1 1 0 error_min error_min 1 }
{ 1 1 0 error_max error_max 1 }
{ 1 1 0 pattern_x pattern_x 1 }
{ 1 1 0 pattern_y pattern_y 1 }
{ 1 1 0 pattern_r pattern_r 1 }
{ 1 1 0 pattern_t pattern_t 1 }
{ 1 1 0 search_x search_x 1 }
{ 1 1 0 search_y search_y 1 }
{ 1 1 0 search_r search_r 1 }
{ 1 1 0 search_t search_t 1 }
{ 2 1 0 key_track key_track 1 }
}
{
 { {curve K x1 1} "track 1" {curve x1001 512.5 516 519.25} {curve x1001 300 301 302} {} {} 1 1 0 {curve x1001 0 0.02 0.04} 0 0.1 -22 -22 22 22 -50 -50 50 50 {curve x1001 1} }
 { {curve K x1 1} "track 2" {curve x1001 812 815 818} {curve x1001 300 300 300} {} {} 1 1 1 {curve x1001 0 0.01 0.01} 0 0.1 -22 -22 22 22 -50 -50 50 50 {curve x1001 1} }
}
}
 transform match-move
 reference_frame 1001
 name Tracker1
}
"""


class TrackerImport(unittest.TestCase):
    def test_tracker4(self):
        _, report, n = convert("Constant { inputs 0 name C }\n" + TRACKER_NK.replace(" inputs 0\n", ""))
        t = n["Tracker1"]
        self.assertEqual(t["type"], "Tracker")
        p = t["params"]
        self.assertEqual(p["transform"], "match-move")
        self.assertEqual(p["reference_frame"], 1001)
        tr = p["tracks"]
        self.assertEqual([x["name"] for x in tr], ["track 1", "track 2"])
        self.assertEqual(tr[0]["keys"][0], [1001.0, 512.5, 300.0, 1.0, 1])
        self.assertEqual(tr[0]["keys"][2][1:4], [519.25, 302.0, 0.96])
        self.assertEqual(tr[1]["R"], True)
        self.assertEqual(tr[1]["S"], True)
        self.assertEqual(tr[0]["pattern"], [-22.0, -22.0, 22.0, 22.0])
        self.assertEqual(tr[0]["search"], [-50.0, -50.0, 50.0, 50.0])

    def test_cornerpin(self):
        _, _, n = convert("Constant { inputs 0 name C }\nCornerPin2D { to1 {10 20} to2 {100 20} to3 {100 90} to4 {10 90} "
                          "from1 {0 0} from2 {90 0} from3 {90 70} from4 {0 70} name CP }\n")
        self.assertEqual(n["CP"]["type"], "CornerPin2D")
        self.assertEqual(n["CP"]["params"]["to2"], [100, 20])


PLANAR_NK = """PlanarTracker {
 inputs 0
 curves {{{v x3f99999a}
  {f 0}
  {n
   {layer Root
    {f 2097152}
    {t x44700000 x44070000}
    {a pt1x {{1001 100} {1002 110}} pt1y {{1001 100} {1002 100}} pt2x {{1001 300} {1002 310}} pt2y {{1001 100} {1002 100}}
       pt3x {{1001 300} {1002 310}} pt3y {{1001 250} {1002 250}} pt4x {{1001 100} {1002 110}} pt4y {{1001 250} {1002 250}} ptref 1001}
    {curvegroup Bezier1 512 bezier
     {{cc
       {f 8192}
       {px x447a4000
        {0 0} {x42c80000 x42c80000} {0 0}
        {0 0} {x43960000 x42c80000} {0 0}
        {0 0} {x43960000 x437a0000} {0 0}
        {0 0} {x42c80000 x437a0000} {0 0}}}
      idem}
     {tx x447a4000 0 0}
     {a tt x40800000}}}}}}
 name PT1
}
"""


class PlanarImport(unittest.TestCase):
    def test_planar_tracker(self):
        _, _, n = convert("Constant { inputs 0 name C }\n" + PLANAR_NK.replace(" inputs 0\n", ""))
        p = n["PT1"]["params"]
        self.assertEqual(n["PT1"]["type"], "PlanarTracker")
        self.assertEqual(p["region"][0], [100.0, 100.0])
        self.assertEqual(p["surface"][2], [300.0, 250.0])
        self.assertEqual(p["reference_frame"], 1001.0)
        self.assertEqual(len(p["keys"]), 2)
        self.assertAlmostEqual(p["keys"][1][3], 10.0)  # h02: moved 10 px right

OFLOW_NK = """Constant {
 inputs 0
 name C
}
OFlow2 {
 input.first 1001
 input.last 1100
 method Blend
 timing2 Frame
 timingFrame2 {{curve x1001 1001 x1100 1050}}
 vectorDetail 0.4
 resampleType Lanczos4
 shutterSamples 3
 shutterTime 0.5
 name OF1
}
Kronos {
 timingOutputSpeed 0.25
 retimedChannels rgba
 name K1
}
"""


class RetimeImport(unittest.TestCase):
    def test_oflow_and_kronos(self):
        _, _, n = convert(OFLOW_NK)
        p = n["OF1"]["params"]
        self.assertEqual(n["OF1"]["type"], "OFlow")
        self.assertEqual(p["method"], "blend")
        self.assertEqual(p["timing"], "frame")
        self.assertEqual(p["input_first"], 1001)
        self.assertEqual(p["input_last"], 1100)
        self.assertEqual(p["frame"]["keys"][1], [1100, 1050])
        self.assertAlmostEqual(p["vector_detail"], 0.4)
        self.assertEqual(p["filter"], "cubic")
        self.assertEqual(p["shutter_samples"], 3)
        k = n["K1"]["params"]
        self.assertEqual(n["K1"]["type"], "OFlow")
        self.assertAlmostEqual(k["speed"], 0.25)
        self.assertEqual(n["K1"]["inputs"], ["OF1"])

    def test_vector_generator(self):
        _, _, n = convert("""Constant {
 inputs 0
 name C
}
VectorGenerator {
 vectorDetail 0.5
 smoothness 0.7
 flickerCompensation true
 name VG1
}
""")
        p = n["VG1"]["params"]
        self.assertEqual(n["VG1"]["type"], "VectorGenerator")
        self.assertAlmostEqual(p["vector_detail"], 0.5)
        self.assertAlmostEqual(p["smoothness"], 0.7)
        self.assertTrue(p["flicker"])
        self.assertEqual(n["VG1"]["inputs"], ["C"])



CAMERA_NK = """Camera2 {
 inputs 0
 translate {{curve x1 0 1 2} 1.5 {curve x1 5 4.5 4}}
 rotate {0 {curve x1 10 12 14} 0}
 focal 35
 haperture 36
 name Cam1
}
Constant {
 inputs 0
 name Plate
}
Card3D {
 inputs 2
 translate {0 1 -3}
 rotate {0 25 0}
 uniform_scale 4
 lens_in_focal 50
 name Card1
}
"""


class CameraImport(unittest.TestCase):
    def test_camera_and_card(self):
        _, report, n = convert(CAMERA_NK)
        cam = n["Cam1"]
        self.assertEqual(cam["type"], "Camera")
        self.assertEqual(cam["params"]["focal"], 35.0)
        self.assertEqual(cam["params"]["haperture"], 36.0)
        self.assertEqual(cam["params"]["translate"][1], 1.5)
        self.assertEqual(cam["params"]["translate"][0]["keys"][2], [3.0, 2.0])
        card = n["Card1"]
        self.assertEqual(card["type"], "Card3D")
        self.assertEqual(card["params"]["rotate"], [0.0, 25.0, 0.0])
        self.assertEqual(card["params"]["uniform_scale"], 4.0)
        self.assertEqual(card["inputs"], ["Plate", "Cam1"])


SCENE_NK = """Axis2 {
 inputs 0
 translate {0 1 0}
 name Ax
}
Constant {
 inputs 0
 name Tex
}
push $cut_paste_input
Card2 {
 inputs 2
 translate {0 0 -3}
 rotate {0 30 0}
 uniform_scale 2
 rows 8
 columns 8
 name Card1
}
Sphere {
 inputs 0
 radius 0.5
 rows 10
 columns 20
 name Ball
}
Scene {
 inputs 2
 name Scn
}
Camera2 {
 inputs 0
 translate {0 1 6}
 focal 40
 name Cam
}
Constant {
 inputs 0
 name BG
}
ScanlineRender {
 inputs 3
 antialiasing medium
 name SR
}
"""


class Scene3DImport(unittest.TestCase):
    def test_scene(self):
        _, report, n = convert(SCENE_NK.replace("push $cut_paste_input\n", "push Ax\npush Tex\n").replace(
            "Constant {\n inputs 0\n name Tex\n}\n", "Constant {\n inputs 0\n name Tex\n}\n"))
        self.assertEqual(n["Card1"]["type"], "Card")
        self.assertEqual(n["Card1"]["params"]["rows"], 8.0)
        self.assertEqual(n["Card1"]["params"]["rotate"], [0.0, 30.0, 0.0])
        self.assertEqual(n["Ball"]["type"], "Sphere")
        self.assertEqual(n["Ball"]["params"]["radius"], 0.5)
        self.assertEqual(n["Scn"]["type"], "Scene")
        self.assertEqual(n["Ax"]["type"], "Axis")
        self.assertEqual(n["SR"]["type"], "ScanlineRender")
        self.assertEqual(n["SR"]["params"]["antialiasing"], "medium")
        self.assertEqual(len(n["SR"]["inputs"]), 3)

FORMATS_NK = """
Read {
 inputs 0
 file scan.####.dpx
 raw true
 name Scan
}
Log2Lin {
 operation log2lin
 blackpoint 90
 name L2L
}
Write {
 file out.####.dpx
 file_type dpx
 datatype "12 bit"
 name OutDpx
}
Write {
 file out.####.tif
 file_type tiff
 datatype "32 bit float"
 compression LZW
 name OutTif
}
Write {
 file out.mov
 file_type mov
 mov64_codec appr
 mov_prores_codec_profile "ProRes 4:4:4:4 XQ 12-bit"
 name OutMov
}
Write {
 file out2.mov
 file_type mov
 mov64_codec AVdn
 mov64_dnxhd_codec_profile "DNxHR HQX 4:2:2 12-bit"
 name OutDnx
}
Write {
 file out.exr
 datatype "32 bit float"
 name OutExr
}
"""


class FormatImport(unittest.TestCase):
    def test_formats(self):
        comp, report, n = convert(FORMATS_NK)
        self.assertEqual(n["L2L"]["type"], "Log2Lin")
        self.assertEqual(n["L2L"]["params"]["blackpoint"], 90)
        self.assertEqual(n["L2L"]["params"]["operation"], "log2lin")
        self.assertEqual(n["OutDpx"]["params"]["bit_depth"], "12")
        self.assertNotIn("datatype", n["OutDpx"]["params"])
        self.assertEqual(n["OutTif"]["params"]["bit_depth"], "32 float")
        self.assertEqual(n["OutTif"]["params"]["tiff_compression"], "lzw")
        self.assertEqual(n["OutMov"]["params"]["codec"], "prores 4444 xq")
        self.assertEqual(n["OutDnx"]["params"]["codec"], "dnxhr hqx")
        self.assertEqual(n["OutExr"]["params"]["datatype"], "float")


CREATIVE_NK = """
Noise {
 inputs 0
 type turbulence
 size {200 100}
 zoffset 0.5
 octaves 6
 gain 0.6
 color {1 0.5 0.25 1}
 name N1
}
Ramp {
 inputs 0
 p0 {0 0}
 p1 {100 0}
 type smooth
 name R1
}
IDistort {
 uv forward
 uv_scale 2
 name ID1
}
Mirror2 {
 flop true
 name M1
}
Glow2 {
 tolerance 0.5
 brightness 2
 size 30
 effect_only true
 name G1
}
TimeEcho {
 framesbehind 4
 framesfade 2
 method plus
 name T1
}
HueShift {
 hue_rotation 45
 saturation 1.2
 name H1
}
"""


class CreativeImport(unittest.TestCase):
    def test_creative(self):
        comp, report, n = convert(CREATIVE_NK)
        self.assertEqual(n["N1"]["type"], "Noise")
        p = n["N1"]["params"]
        self.assertEqual(p["type"], "turbulence")
        self.assertEqual(p["size"], 200)
        self.assertEqual(p["scale"], [1, 0.5])
        self.assertEqual(p["octaves"], 6)
        self.assertEqual(p["color1"], [1, 0.5, 0.25, 1])
        self.assertEqual(n["R1"]["params"]["type"], "smooth")
        self.assertEqual(n["ID1"]["params"]["uv"], "forward.u,forward.v")
        self.assertEqual(n["M1"]["params"]["mode"], "flop")
        self.assertEqual(n["G1"]["params"]["brightness"], [2, 2, 2])
        self.assertTrue(n["G1"]["params"]["effect_only"])
        self.assertEqual(n["T1"]["params"]["frames"], 5)
        self.assertEqual(n["T1"]["params"]["method"], "plus")
        self.assertEqual(n["H1"]["params"]["hue_rotation"], 45)


CG_NK = """
VectorBlur2 {
 uv forward
 scale 0.5
 name VB
}
ZDefocus2 {
 math "far = 0"
 center 0.25
 dof 1
 size 30
 max_size 60
 filter_type bladed
 blades 7
 name ZD
}
Defocus {
 defocus 12
 name DF
}
Cryptomatte {
 cryptoLayer uCryptoMaterial
 matteList "chrome, glass"
 name CM
}
ScanlineRender {
 inputs 0
 samples 10
 shutter 0.5
 shutteroffset start
 name SR
}
"""


class CGImport(unittest.TestCase):
    def test_cg(self):
        comp, report, n = convert(CG_NK)
        self.assertEqual(n["VB"]["params"]["uv"], "forward.u,forward.v")
        self.assertEqual(n["VB"]["params"]["shutter"], 0.5)
        p = n["ZD"]["params"]
        self.assertEqual(p["math"], "1/z")
        self.assertAlmostEqual(p["focus_plane"], 4.0)
        self.assertEqual(p["maximum"], 60)
        self.assertEqual(p["filter_type"], "bladed")
        self.assertEqual(n["DF"]["params"]["defocus"], 12)
        self.assertEqual(n["CM"]["params"]["layer"], "cryptomaterial")
        self.assertEqual(n["CM"]["params"]["matte_list"], "chrome, glass")
        self.assertEqual(n["SR"]["params"]["motionblur"], 10)
        self.assertEqual(n["SR"]["params"]["shutter_offset"], "start")




class ExportRoundTrip(unittest.TestCase):
    """Welder -> .nk (welder nk) -> Welder (this importer) keeps the comp."""

    def roundtrip(self, comp_path):
        exe = os.path.join(ROOT, "build", "strata")
        if not os.path.exists(exe):
            self.skipTest("build/welder not built")
        d = tempfile.mkdtemp()
        nk = os.path.join(d, "x.nk")
        r = subprocess.run([exe, "nk", comp_path, "-o", nk], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        back, _ = nk2strata.convert(nk)
        return json.load(open(comp_path)), back

    def test_groups_example(self):
        a, b = self.roundtrip(os.path.join(ROOT, "examples", "groups.weld"))
        na = {n["name"]: n for n in a["nodes"]}
        nb = {n["name"]: n for n in b["nodes"]}
        self.assertEqual(set(na), set(nb))
        for name, x in na.items():
            y = nb[name]
            self.assertEqual(x["type"], y["type"], name)
            self.assertEqual(x.get("inputs", []), y.get("inputs", []), name)
            self.assertEqual(x.get("group"), y.get("group"), name)
            for k, v in x.get("params", {}).items():
                if k in ("knobs", "color"):
                    continue
                self.assertEqual(y["params"].get(k), v, "%s.%s" % (name, k))
        self.assertEqual(b["format"]["width"], 960)
        self.assertEqual(b["frame_range"], [1.0, 24.0])

    def test_imported_script_survives(self):
        d = tempfile.mkdtemp()
        st = os.path.join(d, "shot.weld")
        comp, _ = nk2strata.convert(os.path.join(ROOT, "examples", "nk", "shot010_comp_v003.nk"))
        json.dump(comp, open(st, "w"))
        a, b = self.roundtrip(st)
        na = {n["name"]: n for n in a["nodes"]}
        nb = {n["name"]: n for n in b["nodes"]}
        self.assertEqual(set(na), set(nb))
        for name, x in na.items():
            self.assertEqual((x["type"], x.get("inputs", []), x.get("group")),
                             (nb[name]["type"], nb[name].get("inputs", []), nb[name].get("group")), name)

    def test_panel_knobs_survive(self):
        """The .nk panel knobs (Read, Transform, Reformat, Merge, Copy, Grade, Write, Node tab) both ways."""
        d = tempfile.mkdtemp()
        src = os.path.join(d, "panel.nk")
        open(src, "w").write(PANEL_NK)
        comp, _ = nk2strata.convert(src)
        st = os.path.join(d, "panel.weld")
        json.dump(comp, open(st, "w"))
        a, b = self.roundtrip(st)
        na = {n["name"]: n for n in a["nodes"]}
        nb = {n["name"]: n for n in b["nodes"]}
        self.assertEqual(set(na), set(nb))
        for name, x in na.items():
            y = nb[name]
            self.assertEqual(x["type"], y["type"], name)
            for k, v in x.get("params", {}).items():
                self.assertEqual(y["params"].get(k), v, "%s.%s" % (name, k))
            for k in ("label", "hide_input", "note_font_size"):
                self.assertEqual(x["ui"].get(k), y["ui"].get(k), "%s ui.%s" % (name, k))
        p = na["Plate"]["params"]
        self.assertEqual((p["frame_mode"], p["frame"], p["after"], p["missing_frames"]), ("start at", "1001", "bounce", "nearest frame"))
        self.assertEqual(na["Plate"]["ui"]["label"], "[value file]")
        self.assertEqual(na["T"]["params"]["filter"], "mitchell")
        m = na["M"]["params"]
        self.assertEqual((m["Achannels_off"], m["Bchannels"], m["video_colorspace"], m["alpha_masking"], m["metadata_from"],
                          m["range_from"], m["inject"], m["fringe"]), (["G"], "depth", True, True, "A", "A", True, True))


PANEL_NK = """Read {
 inputs 0
 file plates/a.####.exr
 format "1920 1080 0 0 1920 1080 1 HD_1080"
 proxy plates/proxy/a.####.exr
 first 1
 last 50
 after bounce
 origfirst 1
 origlast 50
 origset true
 frame_mode "start at"
 frame 1001
 on_error "nearest frame"
 premultiplied true
 auto_alpha true
 label "\[value file]"
 note_font_size 18
 name Plate
}
Transform {
 translate {12 -4}
 rotate 7.5
 scale {1.1 0.9}
 skewX 0.1
 skewY -0.05
 skew_order YX
 center {960 540}
 filter Mitchell
 clamp true
 black_outside false
 motionblur 1
 shutter 0.5
 shutteroffset centred
 useLifetime true
 lifetimeStart 10
 lifetimeEnd 40
 name T
}
Reformat {
 type "to box"
 box_width 1280
 box_height 720
 box_fixed true
 turn true
 pbb true
 filter Lanczos6
 hide_input true
 name R
}
Grade {
 white 2
 add 0.05
 reverse true
 unpremult rgba.alpha
 name G
}
Saturation {
 saturation 0.5
 mode Average
 name S
}
Merge2 {
 inputs 2
 operation plus
 sRGB true
 screen_alpha true
 metainput A
 rangeinput A
 Achannels {rgba.red -rgba.green rgba.blue rgba.alpha}
 Bchannels depth
 inject true
 fringe true
 name M
}
Write {
 file out/a.####.dpx
 file_type dpx
 use_limit true
 first 5
 last 20
 name W
}
"""

if __name__ == "__main__":
    unittest.main()


OCIO_NK = """Root {
 inputs 0
 name /shots/a.nk
 colorManagement OCIO
 OCIO_config aces_1.3
 workingSpaceLUT "ACES - ACEScg"
 int8Lut "Utility - sRGB - Texture"
}
Read {
 inputs 0
 file plate.####.exr
 colorspace "Input - ARRI - V3 LogC (EI800) - Wide Gamut"
 name Plate
}
OCIOColorSpace {
 in_colorspace scene_linear
 out_colorspace "ACES - ACEScct"
 name ToLog
}
OCIOCDLTransform {
 slope {1.1 1 0.9}
 saturation 0.8
 direction inverse
 name Grade
}
Write {
 file out.####.png
 colorspace sRGB
 name Out
}
"""


class ColorImport(unittest.TestCase):
    def test_ocio(self):
        comp, report, n = convert(OCIO_NK)
        c = comp["project"]["color"]
        self.assertEqual(c["working"], "ACES - ACEScg")
        self.assertEqual(c["integer"], "Utility - sRGB - Texture")
        self.assertEqual(n["Plate"]["params"]["colorspace"], "Input - ARRI - V3 LogC (EI800) - Wide Gamut")
        self.assertEqual(n["ToLog"]["type"], "OCIOColorSpace")
        self.assertEqual(n["ToLog"]["params"]["in_colorspace"], "working")
        self.assertEqual(n["ToLog"]["params"]["out_colorspace"], "ACES - ACEScct")
        self.assertEqual(n["Grade"]["params"]["slope"], [1.1, 1.0, 0.9])
        self.assertEqual(n["Grade"]["params"]["direction"], "inverse")
        self.assertEqual(n["Out"]["params"]["colorspace"], "sRGB Encoded Rec.709 (sRGB)")

    def test_nk_default(self):
        comp, report, n = convert("Root {\n inputs 0\n}\nRead {\n inputs 0\n file a.png\n colorspace linear\n name R\n}\n")
        self.assertEqual(comp["project"]["color"]["working"], "Linear Rec.709 (sRGB)")
        self.assertEqual(n["R"]["params"]["colorspace"], "working")
