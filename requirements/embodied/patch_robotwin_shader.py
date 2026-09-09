"""Make RoboTwin's SAPIEN shader selectable via $ROBOTWIN_SHADER.

RoboTwin's ``Base_Task.setup_scene`` hardcodes a 32-samples-per-pixel, 8-bounce
path tracer with no config switch. That is fine on an NVIDIA GPU, but on CDNA
accelerators (MI300/MI350) RADV refuses the device -- ``device 'GFX940' is not
supported by RADV`` -- so the only usable Vulkan driver is the lavapipe CPU
rasterizer. Measured there, the ray-traced path costs ~2.4 s for a single
640x480 frame, versus ~64 env-steps/s with three cameras using the ``default``
rasterization shader.

This rewrite keeps ``rt`` reachable (set ``ROBOTWIN_SHADER=rt``) for parity
checks against ray-traced reference results, but defaults to rasterization so
rollouts are tractable.

Usage: python patch_robotwin_shader.py <path to RoboTwin/envs/_base_task.py>
"""

import io
import re
import sys

OLD = '''        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")'''

NEW = '''        _shader = os.environ.get("ROBOTWIN_SHADER", "default")
        sapien.render.set_camera_shader_dir(_shader)
        if _shader == "rt":
            sapien.render.set_ray_tracing_samples_per_pixel(32)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")'''


def main(path: str) -> int:
    with io.open(path, encoding="utf-8") as handle:
        src = handle.read()

    if "ROBOTWIN_SHADER" in src:
        print(f"[patch_robotwin_shader] already patched: {path}")
        return 0

    if OLD not in src:
        raise SystemExit(
            f"[patch_robotwin_shader] refusing to patch {path}: the expected "
            "set_camera_shader_dir block was not found. RoboTwin upstream has "
            "changed; re-check the shader setup before building."
        )

    src = src.replace(OLD, NEW, 1)
    if not re.search(r"^import os$", src, re.M):
        src = re.sub(r"^import sapien$", "import os\nimport sapien", src, count=1, flags=re.M)

    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(src)

    print(f"[patch_robotwin_shader] patched {path}: shader now $ROBOTWIN_SHADER (default 'default')")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
