"""
Generate a MuJoCo MJCF model of the Inspire RH56 (right) hand from the
reference URDF in renesas-rdk/inspire_rh56_hand_description.

 * Joint frames, axes, limits and mimic ratios come straight from the URDF,
   so kinematics match the real RH56DFX / RH56E2 (6 actuators, 12 joints).
 * URDF <mimic> tags become MuJoCo <equality joint polycoef=...> constraints
   (MuJoCo's URDF importer silently drops mimic joints, so we do it by hand).
 * If the real meshes are present (after `git lfs pull` in the description
   repo), they are used for visuals/collision. Otherwise, capsule/box
   primitives fitted to the link geometry are generated.

Usage:  python models/build_hand.py [--meshdir path/to/meshes/collision]
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "..", "assets", "inspire_rh56", "urdf", "inspire_hand_right_ref.urdf")
OUT = os.path.join(HERE, "inspire_rh56e2_right.xml")

# Fingertip frames (from inspire_hand_right_macro.xacro, "*_tip_joint")
TIPS = {
    "thumb_distal": (0.0202, 0.0140, -0.006),
    "index_intermediate": (-0.0008, 0.045, -0.005),
    "middle_intermediate": (-0.001, 0.048, -0.005),
    "ring_intermediate": (-0.0008, 0.045, -0.005),
    "pinky_intermediate": (-0.0008, 0.037, -0.005),
}

# The 6 physical actuators of the RH56, in the order of the Inspire serial API
# (angleSet[0..5] = little, ring, middle, index, thumb bend, thumb rotation).
ACTUATORS = [
    ("little", "pinky_proximal_joint", 1.0),
    ("ring", "ring_proximal_joint", 1.0),
    ("middle", "middle_proximal_joint", 1.0),
    ("index", "index_proximal_joint", 1.0),
    ("thumb_bend", "thumb_proximal_pitch_joint", 1.2),
    ("thumb_rot", "thumb_proximal_yaw_joint", 0.8),
]


def rpy_to_quat(r, p, y):
    """URDF fixed-axis roll-pitch-yaw -> MuJoCo quaternion (w x y z)."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def f(v):
    return " ".join(f"{x:.6g}" for x in v)


def is_real_mesh(path):
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as fh:
        return not fh.read(64).startswith(b"version https://git-lfs")


def primitive_geoms(link, child_offsets):
    """Capsules fitted between a link's joint and its child joint / tip."""
    g = []
    if link == "thumb_proximal_base":
        g.append(f'<geom class="rh_col" type="sphere" size="0.009" pos="-0.004 0.001 -0.0076"/>')
    elif link == "thumb_proximal":
        cx, cy, _ = child_offsets[0]
        g.append(f'<geom class="rh_col" type="capsule" size="0.0095" fromto="0.004 0.004 -0.008 {cx:.4f} {cy:.4f} -0.008"/>')
    elif link in ("thumb_intermediate", "thumb_distal"):
        end = child_offsets[0] if child_offsets else TIPS[link]
        r = 0.0085 if link == "thumb_intermediate" else 0.0075
        ex, ey = end[0], end[1]
        n = math.hypot(ex, ey)
        ex, ey = ex * (1 - r / n), ey * (1 - r / n)
        g.append(f'<geom class="rh_col" type="capsule" size="{r}" fromto="0 0 -0.0065 {ex:.4f} {ey:.4f} -0.0065"/>')
    elif link.endswith("_proximal"):
        cx, cy, _ = child_offsets[0]
        g.append(f'<geom class="rh_col" type="capsule" size="0.0078" fromto="0 0.004 -0.0058 {cx:.4f} {cy:.4f} -0.0058"/>')
    elif link.endswith("_intermediate"):
        tx, ty, _ = TIPS[link]
        g.append(f'<geom class="rh_pad" type="capsule" size="0.0068" fromto="0 0 -0.005 {tx:.4f} {ty - 0.0068:.4f} -0.005"/>')
    return g


def build(meshdir=None):
    root = ET.parse(URDF).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    links = {l.get("name"): l for l in root.findall("link")}

    use_mesh = meshdir is not None and all(
        is_real_mesh(os.path.join(meshdir, f"right_{n}.obj"))
        for n in ("index_proximal", "thumb_distal"))

    assets, equalities = [], []

    def inertial(link):
        i = links[link].find("inertial")
        if i is None:
            return ""
        m = float(i.find("mass").get("value"))
        pos = [float(x) for x in i.find("origin").get("xyz").split()]
        I = i.find("inertia")
        full = [float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        return f'<inertial pos="{f(pos)}" mass="{m:.6g}" fullinertia="{f(full)}"/>'

    def body(link, indent):
        pad = "  " * indent
        out = []
        j = joints[link]
        o = j.find("origin")
        pos = [float(x) for x in o.get("xyz").split()]
        rpy = [float(x) for x in o.get("rpy").split()]
        out.append(f'{pad}<body name="{link}" pos="{f(pos)}" quat="{f(rpy_to_quat(*rpy))}">')
        out.append(f"{pad}  {inertial(link)}")
        if j.get("type") == "revolute":
            ax = j.find("axis").get("xyz")
            lim = j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            out.append(f'{pad}  <joint name="{j.get("name")}" axis="{ax}" range="{lo:.5g} {hi:.5g}"/>')
            mim = j.find("mimic")
            if mim is not None:
                mul = float(mim.get("multiplier", 1)); off = float(mim.get("offset", 0))
                equalities.append(
                    f'<joint joint1="{j.get("name")}" joint2="{mim.get("joint")}" '
                    f'polycoef="{off:.6g} {mul:.6g} 0 0 0" solref="0.005 1"/>')
        kids = [c for c in children.get(link, []) if c.get("type") != "fixed"]
        child_offs = [tuple(float(x) for x in c.find("origin").get("xyz").split()) for c in kids]
        if use_mesh and os.path.isfile(os.path.join(meshdir, f"right_{link}.obj")):
            assets.append(f'<mesh name="rh_{link}" file="{os.path.abspath(os.path.join(meshdir, f"right_{link}.obj"))}"/>')
            out.append(f'{pad}  <geom class="rh_col" type="mesh" mesh="rh_{link}"/>')
        else:
            for g in primitive_geoms(link, child_offs):
                out.append(f"{pad}  {g}")
        if link in TIPS:
            out.append(f'{pad}  <site name="{link.split("_")[0]}_tip" pos="{f(TIPS[link])}" size="0.004" rgba="1 0 0 1" group="4"/>')
        for c in kids:
            out += body(c.find("child").get("link"), indent + 1)
        out.append(f"{pad}</body>")
        return out

    # hand_base_link collision primitives straight from the URDF
    base_geoms = []
    for c in links["hand_base_link"].findall("collision"):
        o = c.find("origin")
        pos = o.get("xyz"); q = rpy_to_quat(*[float(x) for x in o.get("rpy").split()])
        geo = c.find("geometry")
        if geo.find("box") is not None:
            s = [float(x) / 2 for x in geo.find("box").get("size").split()]
            base_geoms.append(f'<geom class="rh_palm" type="box" size="{f(s)}" pos="{pos}" quat="{f(q)}"/>')
        elif geo.find("cylinder") is not None:
            cy = geo.find("cylinder")
            base_geoms.append(f'<geom class="rh_palm" type="cylinder" size="{cy.get("radius")} {float(cy.get("length"))/2}" pos="{pos}" quat="{f(q)}"/>')

    fingers = []
    for c in children["hand_base_link"]:
        fingers += body(c.find("child").get("link"), 4)

    hb_q = rpy_to_quat(-1.5707963, 0, 3.14159)
    acts = []
    for name, jn, frc in ACTUATORS:
        jel = joints[[k for k, v in joints.items() if v.get("name") == jn][0]]
        hi = float(jel.find("limit").get("upper"))
        acts.append(f'<position name="{name}" joint="{jn}" ctrlrange="0 {hi:.4g}" kp="3" kv="0.05" forcerange="-{frc} {frc}"/>')

    xml = f"""<!-- AUTO-GENERATED by models/build_hand.py from the Inspire RH56 reference URDF. -->
<mujoco model="inspire_rh56e2_right">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <default class="rh">
      <joint damping="0.02" armature="0.0005" frictionloss="0.002"/>
      <default class="rh_col">
        <geom rgba="0.15 0.15 0.17 1" friction="1.5 0.02 0.001" condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" margin="0.0005"/>
        <default class="rh_pad"><geom rgba="0.25 0.25 0.28 1" friction="2.0 0.02 0.001"/></default>
        <default class="rh_palm"><geom rgba="0.82 0.82 0.85 1"/></default>
      </default>
    </default>
  </default>
  <asset>
    {chr(10).join('    ' + a for a in assets)}
  </asset>
  <worldbody>
    <!-- 'rh_base' = URDF 'base' frame: +Z along the fingers, +X out of the palm, +Y toward the index side -->
    <body name="rh_base" childclass="rh">
      <site name="grasp_site" pos="0.042 0.0 0.115" size="0.006" rgba="0 1 0 0.6" group="4"/>
      <site name="tcp" pos="0.075 0.015 0.155" quat="{f(rpy_to_quat(0, 0.785398, 0))}" size="0.005" group="4"/>
      <body name="hand_base_link" quat="{f(hb_q)}">
        {inertial("hand_base_link")}
{chr(10).join('        ' + g for g in base_geoms)}
{chr(10).join(fingers)}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="hand_base_link" body2="thumb_proximal_base"/>
    <exclude body1="hand_base_link" body2="thumb_proximal"/>
    <exclude body1="hand_base_link" body2="index_proximal"/>
    <exclude body1="hand_base_link" body2="middle_proximal"/>
    <exclude body1="hand_base_link" body2="ring_proximal"/>
    <exclude body1="hand_base_link" body2="pinky_proximal"/>
  </contact>
  <equality>
{chr(10).join('    ' + e for e in equalities)}
  </equality>
  <actuator>
{chr(10).join('    ' + a for a in acts)}
  </actuator>
</mujoco>
"""
    with open(OUT, "w") as fh:
        fh.write(xml)
    print(f"wrote {OUT}  ({'real meshes' if use_mesh else 'primitive collision geometry'})")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshdir", default=os.path.join(HERE, "..", "assets", "inspire_rh56", "meshes", "collision"))
    build(ap.parse_args().meshdir)
