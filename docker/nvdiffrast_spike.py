"""Proof-of-concept: EGL-free GPU rasterization in Docker via nvdiffrast's CUDA backend.

The headless-GPU problem on this kind of box (WSL2, no /dev/dri): OpenGL/EGL offscreen contexts
fail, which rules out Habitat-Sim, Isaac Sim and CARLA (all need an EGL/Vulkan display) for
rendering SLAM worlds. But CUDA *compute* works. nvdiffrast's `RasterizeCudaContext` rasterizes
through CUDA with NO OpenGL/EGL context — so it (and Blender's Cycles, used by Rung 5) can render
on the GPU here where GL-based simulators cannot.

This script just proves the path inside a container; run it with docker/run_nvdiffrast_spike.sh.
A full GPU rasterized world-rung would extend this: project a textured mesh + camera trajectory,
read depth from the rast z/w channel, and emit the same frames.npz + gt_poses.csv contract."""
import numpy as np
import torch
import nvdiffrast.torch as dr

print("cuda available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
glctx = dr.RasterizeCudaContext()          # CUDA rasterization, no OpenGL/EGL context
pos = torch.tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [0.0, 0.8, 0, 1]]],
                   dtype=torch.float32, device="cuda")
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[64, 64])
covered = int((rast[..., 3] > 0).sum().item())
print("rasterized triangle:", tuple(rast.shape), "covered px", covered)
assert covered > 100
print("NVDIFFRAST CUDA OK (EGL-free GPU rasterization works in Docker)")
