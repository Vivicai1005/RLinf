# [Roadmap] RLinf on AMD ROCm - 26Q4

## Embodied AI

#### RoboTwin 2.0 — Radeon PRO W7900 (gfx1100, 1 node × 8 GPUs) — P0
- [ ] Validate OpenVLA-OFT + GRPO training and accuracy on representative RoboTwin tasks.
- [ ] Enable Lingbot-VLA + GRPO fine-tuning on ROCm.
- [ ] Enable PPO fine-tuning of π₀ and π₀.₅ on ROCm.
- [ ] Publish a ROCm RoboTwin Docker image and update the deployment documentation.
- [ ] Add RoboTwin ROCm CI coverage.

#### LIBERO — Instinct MI300 (gfx942) · Radeon PRO W7900 (gfx1100), 1 node × 8 GPUs — P1
- [ ] Validate the existing ROCm LIBERO Docker image on both Instinct MI300 and Radeon PRO W7900.
- [ ] Validate OpenVLA-OFT + GRPO/PPO training and accuracy on ROCm.
- [ ] Enable π₀ / π₀.₅ fine-tuning using supported RL algorithms.
      
#### ManiSkill (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs) - P1
- [ ] Validate OpenVLA-OFT + PPO training and accuracy on ManiSkill3.

#### MetaWorld  (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs) - P2
- [ ] Enable π₀ / π₀.₅ + PPO fine-tuning on MetaWorld MT50.
- [ ] Enable OpenVLA-OFT + GRPO fine-tuning on MetaWorld MT50.

#### RoboCasa365 (Instinct MI300 (`gfx942`) · Radeon PRO W7900 (`gfx1100`), 1 node × 8 GPUs) - P2
- [ ] Enable π₀ + PPO fine-tuning on RoboCasa365.


## World Models

#### Wan — Instinct MI300 (gfx942, 1 node × 8 GPUs) — P2
- [ ] Enable OpenVLA-OFT + GRPO fine-tuning with the Wan world model on ROCm. Validate training and accuracy on LIBERO Spatial, Object, and Goal.

#### OpenSora — Instinct MI300 (gfx942, 1 node × 8 GPUs) — P2
- [ ] Enable OpenVLA-OFT + GRPO fine-tuning with the OpenSora world model on ROCm. Validate training and accuracy on LIBERO Spatial and Object.





