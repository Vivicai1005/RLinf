## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning image and the various embodied images. Use the `BUILD_TARGET` build argument to select which image to build:

- `reason` — math reasoning image
- `embodied-<env>` — embodied image for a specific environment (and optionally a specific model when multiple model flavors exist for the same env)

To build the Docker image, run the following command **in the RLinf root directory**:

```shell
export BUILD_TARGET=reason # or one of the embodied-* targets defined in the Dockerfile
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

### Available `BUILD_TARGET` values

Each `BUILD_TARGET` maps to a build stage in [`Dockerfile`](Dockerfile). To see the full, up-to-date list of targets and the venvs each one installs, look at the stage names (`FROM ... AS <target>-image`) and the `install.sh` invocations inside them — the Dockerfile is the source of truth, so this README does not duplicate the list.

### Additional build arguments

- `PLATFORM` (default `nvidia`) — hardware platform: `nvidia` (CUDA), `amd` (ROCm), `ascend` (CANN), or `musa` (Moore Threads). Selects the base image and is also recorded as `RLINF_PLATFORM` in the final image. The `embodied-franka` target ignores `PLATFORM` and always uses a plain `ubuntu:20.04` base.
- Per-platform runtime versions: `CUDA_VER`, `ROCM_VER`, `ROCM_ARCHS`, `ROCM_TORCH_VER`, `CANN_VER`, `MUSA_VER`, `UBUNTU_VER`, `AMD_UBUNTU_VER`. Override any of these to bump versions without changing the rest of the build. For a fully custom base, set `NVIDIA_BASE_IMAGE`, `AMD_BASE_IMAGE`, `ASCEND_BASE_IMAGE`, or `MUSA_BASE_IMAGE` directly.
- `UV_PATH` — where RLinf's venvs are created. Defaults to `/opt/venv`, except on `PLATFORM=amd` where it is `/opt/rlinf-venv` because the ROCm base image already owns `/opt/venv`.
- `NO_MIRROR` — set to `1` to skip the USTC apt/pypi mirror rewrites (recommended outside of mainland China).

Example with non-default args:

```shell
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-metaworld \
    --build-arg PLATFORM=nvidia \
    --build-arg CUDA_VER=12.4.1 \
    --build-arg NO_MIRROR=1 \
    -t rlinf:embodied-metaworld .
```

### Building for AMD Radeon / ROCm

`PLATFORM=amd` builds on `rocm/pytorch`, which already carries a torch built
against the image's ROCm. `install.sh` reuses that build rather than resolving a
wheel: the vendor local-version tag exists on no public index, and replacing
torch would break the triton/flash-attn kernels compiled against it. Because
those images own `/opt/venv`, RLinf's own venvs are placed in `/opt/rlinf-venv`
instead.

The defaults target RDNA3 (`gfx1100`: W7900/W7900D, RX 7900 XT(X)). Set
`ROCM_ARCHS=gfx90a;gfx942` for CDNA (MI200/MI300) — that path additionally needs
`AMD_VULKAN_ICD_FILE` pointed at lavapipe, since RADV refuses CDNA parts and
SAPIEN would otherwise find no Vulkan device.

`ROCM_VER` and `ROCM_TORCH_VER` together select the base image tag, so they have
to name a tag that actually exists. Prefer a ROCm no newer than the host driver.

```shell
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-robotwin \
    --build-arg PLATFORM=amd \
    -t rlinf:embodied-robotwin-rocm .
```

Run it with the GPUs passed through. There is no `--gpus all` equivalent on
ROCm; the KFD and DRI device nodes are handed over directly:

```shell
docker run -it --rm --ipc=host --shm-size=100g --network host \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    rlinf:embodied-robotwin-rocm bash
```

Only the `lingbotvla` venv is installed on AMD: `openpi` pins `jax[cuda12]`, and
`openvla-oft` is not validated on ROCm. RoboTwin itself is baked into the image
at `$ROBOTWIN_PATH`; its multi-GB assets are still a runtime download via
`script/_download_assets.sh`.

### Building for Moore Threads (MUSA)

`PLATFORM=musa` builds on top of the Moore Threads training suite image
(`registry.mthreads.com/mcctest/ai/training-suite:$MUSA_VER`), which already
carries a MUSA-built torch plus `torch-musa`. `install.sh` therefore installs no
torch of its own — it creates the venv with `--system-site-packages` on the
image's interpreter and skips every CUDA-only package (flash-attn, apex, and the
vLLM/SGLang kernels). The `embodied-maniskill_libero` target builds the subset of
models that need none of them (`openpi` and `gr00t`) when `PLATFORM=musa`. Build
and run it with the `mthreads` container runtime:

Build with BuildKit — the legacy builder resolves every `FROM` in the
Dockerfile, including the CUDA and ROCm bases on Docker Hub that a MUSA host
often cannot reach.

```shell
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg PLATFORM=musa \
    -t rlinf:embodied-maniskill_libero .

docker run -it --runtime=mthreads --ipc=host --shm-size=100g \
    -e MTHREADS_VISIBLE_DEVICES=all \
    rlinf:embodied-maniskill_libero bash
```

# Using the Docker Image

The built Docker image contains one or more Python virtual environments (venvs) under `/opt/venv/`. Which venvs are present, and which one is activated by default in new shells, depends on the `BUILD_TARGET` — see the corresponding build stage in the [`Dockerfile`](Dockerfile).

To switch between venvs, use the built-in `switch_env` script:

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env openpi, etc.
```