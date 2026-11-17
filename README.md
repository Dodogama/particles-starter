# particles-starter

GPU particle system with **PyOpenGL compute shaders (GL 4.3+)** and **pygame-ce** for the window and input.
1M particles live in one SSBO. A compute shader integrates them, and the same buffer is drawn as
`GL_POINTS` with additive blending. Particle data never comes back to the CPU.

## Setup / run
```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python particles.py [--count 4000000]
.venv/bin/python particles.py --shot out.png   # 240 fixed-step frames, headless screenshot
```
Controls: hold LMB to attract, RMB to repel, R to reset, Space to pause, Esc to quit.
