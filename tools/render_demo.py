"""Render website demo clips of a trained CR-Open policy (presentation media, not evaluation data).

One env, one family forced through the env config (cfg.families = (F,)), full gate thresholds, the deterministic
policy of an rl_games checkpoint (same agent config as train_ppo.py). An RTX camera sensor added to the scene
(CameraCfg "demo_cam", needs --enable_cameras; IsaacLab 3 dropped render_mode "rgb_array") records env 0 at the
control rate (30 Hz). Episodes are tried until one succeeds (at most
--attempts); the log records every attempt, so the selection is disclosed. Frames -> H.264 MP4 (ffmpeg) + poster JPG.
Runs on the GPU box in a scratch copy of the cropen code (never the E3 checkout).

Usage: python -B render_demo.py --cropen-dir <copy>/cropen --ckpt <ckpt.pt> --family M3 --out <dir> [--still]
       [--eye x y z] [--lookat x y z] [--res 1280 720] [--attempts 8] [--seed 100] [--tag name]
"""
import argparse
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cropen-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--family", required=True, choices=("M1", "M2", "M3", "M6"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--eye", type=float, nargs=3, default=(0.30, 0.22, 0.50))
    p.add_argument("--lookat", type=float, nargs=3, default=(0.02, -0.04, 0.34))
    p.add_argument("--res", type=int, nargs=2, default=(1280, 720))
    p.add_argument("--still", action="store_true", help="save one frame after a few steps (camera check) and stop")
    p.add_argument("--pretty", action="store_true",
                   help="presentation look: procedural wood table (UsdPreviewSurface texture on a UV quad), amber bottle and white cap, dark "
                        "hand with white fingertips, warm key light with soft shadows, ground plane hidden. Visual "
                        "only: no collider, no change to physics or to the policy's inputs")
    p.add_argument("--focal", type=float, default=18.0)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    p.set_defaults(headless=True, enable_cameras=True)
    a = p.parse_args()
    app = AppLauncher(a).app
    try:
        run(a)
    finally:
        app.close()


def run(a):
    import cv2
    import numpy as np
    sys.path.insert(0, str(a.cropen_dir))
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    import train_ppo
    from env_cropen import CROpenEnv, CROpenEnvCfg

    tag = a.tag or a.family
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = CROpenEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = a.seed
    cfg.families = (a.family,)
    cfg.gate_curriculum = True                       # the recipe's flags; full_gates overrides the thresholds below
    cfg.gate_up, cfg.gate_scale_step, cfg.gate_scale_init = 0.3, 0.05, 0.25
    cfg.r_unlock = 2.0
    cfg.obs_history = 8
    cfg.observation_space = 76 * 8
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCam", update_period=0, width=a.res[0], height=a.res[1], data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=a.focal, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.01, 20.0)))
    env = CROpenEnv(cfg)
    if a.pretty:
        dress_scene(a, sim_utils)
    cam = env.scene["demo_cam"]
    org = env.scene.env_origins[0:1].to(torch.float32)
    cam.set_world_poses_from_view(torch.tensor([a.eye], device=org.device) + org,
                                  torch.tensor([a.lookat], device=org.device) + org)

    def grab():
        rgb = cam.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        return np.ascontiguousarray(rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    targs = types.SimpleNamespace(num_envs=1, horizon=16, minibatches=2, lr=5e-4, epochs=1, seed=a.seed)
    agent = train_ppo.agent_config(targs, a.out, cfg.state_space)
    pp = agent["params"]
    wrapper = RlGamesVecEnvWrapper(env, "cuda:0", pp["env"]["clip_observations"], pp["env"]["clip_actions"])
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", dict(vecenv_type="IsaacRlgWrapper", env_creator=lambda **kw: wrapper))
    runner = Runner()
    runner.load(agent)
    runner.reset()
    player = runner.create_player()
    player.restore(str(a.ckpt))
    max_ep = int(env.max_episode_length)
    log = []
    env.full_gates = True
    obs = player.env_reset(wrapper)
    player.get_batch_size(obs, 1)
    for k in range(a.attempts):
        env.full_gates = True
        obs = player.env_reset(wrapper)            # forced reset: a fresh episode at the full thresholds
        player.reset()
        env.first_done[:] = False
        env.recording = True
        frames, steps = [], 0
        for steps in range(1, max_ep + 3):
            obs = player.env_step(wrapper, player.get_action(obs, is_deterministic=True))[0]
            frames.append(grab())
            if a.still and len(frames) >= 10:
                cv2.imwrite(str(a.out / f"{tag}_still.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
                print("still", a.out / f"{tag}_still.png", frames[-1].shape)
                return
            if bool(env.first_done[0]):
                break
        ok = bool(env.first_success[0])
        log.append(dict(attempt=k, steps=steps, seconds=round(steps / 30, 2), success=ok, frames=len(frames)))
        print(json.dumps(log[-1]), flush=True)
        if ok and frames:
            tmp = a.out / f"_frames_{tag}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            for i, f in enumerate(frames):
                cv2.imwrite(str(tmp / f"f_{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            mp4 = a.out / f"{tag}.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(tmp / "f_%05d.png"),
                            "-vf", "tpad=stop_mode=clone:stop_duration=1,format=yuv420p", "-c:v", "libx264",
                            "-crf", "23", "-preset", "medium", "-movflags", "+faststart", str(mp4)], check=True,
                           stdin=subprocess.DEVNULL)   # without this ffmpeg stops (SIGTTIN) when run under tmux
            pick = frames[int(0.6 * (len(frames) - 1))]
            cv2.imwrite(str(a.out / f"{tag}-poster.jpg"), cv2.cvtColor(pick, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            shutil.rmtree(tmp, ignore_errors=True)
            break
    env.recording = False
    (a.out / f"{tag}_render_log.json").write_text(json.dumps(dict(
        family=a.family, ckpt=str(a.ckpt), seed=a.seed, eye=a.eye, lookat=a.lookat, attempts=log,
        note="presentation media: family forced via cfg.families; first successful attempt kept; all attempts listed"),
        indent=1))


def wood_texture(path, size=2048, seed=3):
    """procedural wood (numpy): long grain along x, gentle growth rings, fine fibres; low contrast; saved as PNG"""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    def smooth_noise(h, w, octaves=4):
        acc = np.zeros((size, size), np.float32)
        for o in range(octaves):
            n = rng.random((max(2, h * 2 ** o), max(2, w * 2 ** o))).astype(np.float32)
            acc += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC) / 2 ** o
        return acc / 1.9
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    warp = smooth_noise(6, 2) * 0.8                       # slow variation, stretched along x
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (y * 14.0 + warp * 2.5))
    rings = rings ** 3
    fibres = cv2.resize(rng.random((size, 48)).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    fibres = cv2.GaussianBlur(fibres, (0, 0), 1.2)
    blotch = smooth_noise(3, 3, 3)
    t = np.clip(0.45 * rings + 0.25 * fibres + 0.30 * blotch, 0, 1)[..., None]
    light = np.array([190, 150, 108], np.float32)       # RGB, oak
    dark = np.array([140, 98, 62], np.float32)
    img = (light + (dark - light) * t).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(path)


def dress_scene(a, sim_utils):
    """presentation materials and lights on env 0 (visual only)"""
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    stage = omni.usd.get_context().get_stage()
    env0 = "/World/envs/env_0"
    ground = stage.GetPrimAtPath("/World/ground")
    if ground.IsValid():
        UsdGeom.Imageable(ground).MakeInvisible()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(env0 + "/Closure")).ComputeAlignedRange()
    zmin, cmid = rng.GetMin()[2], (rng.GetMin() + rng.GetMax()) / 2
    tex = wood_texture(a.out / "wood.png")
    # table top: a quad with explicit UVs and a UsdPreviewSurface reading the wood texture (repeat 3 x 3)
    half, ztop = 2.5, float(zmin) - 0.0005
    cx, cy = float(cmid[0]), float(cmid[1])
    mesh = UsdGeom.Mesh.Define(stage, "/World/Table")
    mesh.CreatePointsAttr([Gf.Vec3f(cx - half, cy - half, ztop), Gf.Vec3f(cx + half, cy - half, ztop),
                           Gf.Vec3f(cx + half, cy + half, ztop), Gf.Vec3f(cx - half, cy + half, ztop)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(2.5, 0), Gf.Vec2f(2.5, 2.5), Gf.Vec2f(0, 2.5)])   # ~2 m per tile
    mat = UsdShade.Material.Define(stage, "/World/Looks/Wood")
    pbr = UsdShade.Shader.Define(stage, "/World/Looks/Wood/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    rd = UsdShade.Shader.Define(stage, "/World/Looks/Wood/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, "/World/Looks/Wood/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    mats = {
        "Amber": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.17, 0.03), roughness=0.18, metallic=0.0),
        "CapWhite": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.93, 0.91), roughness=0.45, metallic=0.0),
        "HandDark": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.09, 0.09, 0.10), roughness=0.35, metallic=0.4),
        "TipWhite": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.90, 0.88), roughness=0.7, metallic=0.0),
    }
    for name, m in mats.items():
        m.func(f"/World/Looks/{name}", m)
    strong = UsdShade.Tokens.strongerThanDescendants
    def bind(path, mat):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(f"/World/Looks/{mat}")),
                                                         bindingStrength=strong)
    for child in stage.GetPrimAtPath(env0 + "/Robot").GetChildren():
        n = child.GetName()
        if n == "allegro_mount":                     # the mounting block: hidden for the presentation
            UsdGeom.Imageable(child).MakeInvisible()
            continue
        bind(child.GetPath().pathString, "TipWhite" if n.endswith("_link_3") or "tip" in n else "HandDark")
    closure_parts = {}
    cap_links = ("shell", "inner", "tab_l", "tab_r", "carrier")
    for prim in Usd.PrimRange(stage.GetPrimAtPath(env0 + "/Closure")):
        n = prim.GetName()
        if n == "neck":                     # the neck's own geometry only: the cap links are nested below it
            for c in prim.GetChildren():
                if c.GetName() not in cap_links:
                    bind(c.GetPath().pathString, "Amber")
                    closure_parts.setdefault("Amber", []).append(c.GetPath().pathString)
        elif n in cap_links:
            bind(prim.GetPath().pathString, "CapWhite")
            closure_parts.setdefault("CapWhite", []).append(prim.GetPath().pathString)
    print("closure bindings", closure_parts)
    dome = stage.GetPrimAtPath("/World/Light")
    if dome.IsValid():
        for attr, val in (("inputs:intensity", 450.0), ("inputs:color", Gf.Vec3f(0.92, 0.95, 1.0))):
            at = dome.GetAttribute(attr)
            if at:
                at.Set(val)
    key = sim_utils.DistantLightCfg(intensity=1700.0, angle=4.0, color=(1.0, 0.95, 0.88))
    key.func("/World/KeyLight", key, orientation=(0.2706, 0.2706, 0.0, 0.9239))   # xyzw: ~45 deg tilt
    print("dressed scene: table top z", round(float(zmin), 4), "robot links",
          [c.GetName() for c in stage.GetPrimAtPath(env0 + "/Robot").GetChildren()][:30],
          "closure links", [c.GetName() for c in stage.GetPrimAtPath(env0 + "/Closure").GetChildren()])


if __name__ == "__main__":
    main()
