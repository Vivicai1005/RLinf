# [Roadmap] RLinf on AMD ROCm 

## Benchmarks for Embodied AI

#### RoboTwin (Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs)
- [ ] **P0** - Publish the ROCm RoboTwin image. The `embodied-robotwin` stage
  builds with `PLATFORM=amd`, but `docker-build.yml` has no AMD job and every
  build is `push: false`. *Done when* a tagged image is pullable and the docs
  drop the build-from-checkout instructions.
- [ ] **P0** - Enable GRPO fine-tuning of Lingbot-VLA on RoboTwin 2.0 dual-arm
  tasks on ROCm, and validate accuracy. 
- [ ] **P0** - Cover RoboTwin on the ROCm CI runner. A `runs-on: rocm` job exists
  for LIBERO + OpenVLA-OFT, but both RoboTwin jobs still install without
  `--platform amd`. 
- [ ] **P1** - Enable GRPO and PPO fine-tuning of OpenVLA-OFT on RoboTwin 2.0
  dual-arm tasks on ROCm, and validate accuracy.

#### LIBERO (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs)
- [ ] **P1** - Validate the accuracy of OpenVLA-OFT, π₀/π₀.₅, and GR00T N1.5 on
  LIBERO on ROCm.
- [ ] **P1** - Support Radeon PRO W7900 (`gfx1100`) on LIBERO. The published
  `agentic-rlinf0.4-maniskill_libero-rocm7.2` image targets CDNA only
  (`ROCM_ARCHS=gfx90a;gfx942`).
- [ ] **P2** - Validate LIBERO-Pro and LIBERO-Plus on ROCm.

#### ManiSkill (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs)
- [ ] **P1** - validate accuracy of `maniskill_ppo_openvlaoft` on ROCm.
- [ ] **P1** - Support Radeon PRO W7900 (`gfx1100`) on ManiSkill, where RADV
  renders SAPIEN on the GPU instead of lavapipe on the CPU as on CDNA.

#### MetaWorld  (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs)
- [ ] **P2** - Enable PPO fine-tuning of π₀ and π₀.₅ on MetaWorld MT50 on ROCm,
  and validate accuracy.
- [ ] **p2** - Cover MetaWorld on the ROCm CI runner. The
  `embodied-openpi-metaworld-test` and `embodied-openvlaoft-metaworld-test` jobs
  still install without `--platform amd`.
- [ ] **P2** - Enable GRPO fine-tuning of OpenVLA-OFT on MetaWorld MT50 on ROCm.

#### RoboCasa365 (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs)
- [ ] **P2** - Enable PPO fine-tuning of π₀ on RoboCasa365 on ROCm, and validate
  accuracy on the `atomic_seen` slice.


## World Models

#### Wan (Instinct MI300 (`gfx942`), 1 node × 8 GPUs)
- [ ] **P2** - Enable GRPO fine-tuning of OpenVLA-OFT with the Wan world model on
  ROCm, and validate accuracy on LIBERO Spatial, Object, and Goal.

#### OpenSora (Instinct MI300 (`gfx942`), 1 node × 8 GPUs)
- [ ] **P2** - Enable GRPO fine-tuning of OpenVLA-OFT with the OpenSora world
  model on ROCm, and validate accuracy on LIBERO Object and Spatial.

## Framework & System Optimizations
- [ ] **P1** - Support the SGLang rollout backend for VLA/WAM models on ROCm.





