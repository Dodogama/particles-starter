"""GPU particle system: PyOpenGL compute shaders (GL 4.3+) + pygame-ce window/input.

Particles live in one SSBO that the compute shader updates and the vertex shader reads
directly as vertex attributes -- no CPU round-trip.

Controls: hold LMB = attract to mouse, RMB = repel, R = reset, Space = pause, Esc = quit.
Usage:  python particles.py [--count 1000000] [--shot out.png]
"""
import argparse, ctypes, time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

CS = """#version 430
layout(local_size_x = 256) in;
struct P { vec4 pos; vec4 vel; };
layout(std430, binding = 0) buffer Particles { P p[]; };
uniform float dt; uniform vec2 attractor; uniform float strength; uniform uint count; uniform float aspect;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    vec2 pos = p[i].pos.xy, vel = p[i].vel.xy;
    vec2 d = attractor - pos;
    float r2 = dot(d, d) + 0.02;
    vec2 acc = strength * d / (r2 * sqrt(r2)) * 0.15        // inverse-square pull
             + vec2(-d.y, d.x) / (r2 + 0.3) * 0.6;           // swirl around the attractor
    vel = (vel + acc * dt) * exp(-0.5 * dt);                 // damping
    pos += vel * dt;
    if (abs(pos.x) > aspect) { pos.x = sign(pos.x) * aspect; vel.x *= -0.5; }
    if (abs(pos.y) > 1.0)    { pos.y = sign(pos.y);          vel.y *= -0.5; }
    p[i].pos.xy = pos; p[i].vel.xy = vel;
}
"""

VS = """#version 430
layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aVel;
uniform float aspect;
out vec4 vColor;
void main() {
    float s = clamp(length(aVel.xy) * 0.6, 0.0, 1.0);
    vColor = vec4(mix(vec3(0.10, 0.25, 1.0), vec3(1.0, 0.55, 0.15), s), 0.12);
    gl_Position = vec4(aPos.x / aspect, aPos.y, 0.0, 1.0);
}
"""

FS = """#version 430
in vec4 vColor; out vec4 frag;
void main() { frag = vColor; }
"""


def seed(count, aspect, rng=np.random.default_rng(0)):
    """Disk of particles with tangential velocity; layout matches struct P (2x vec4)."""
    r = np.sqrt(rng.random(count)) * 0.8
    a = rng.random(count) * 2 * np.pi
    data = np.zeros((count, 8), np.float32)
    data[:, 0], data[:, 1] = r * np.cos(a) * aspect, r * np.sin(a)
    data[:, 3] = 1.0
    data[:, 4], data[:, 5] = -np.sin(a) * r * 0.5, np.cos(a) * r * 0.5
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1_000_000)
    ap.add_argument("--shot", help="simulate 240 fixed-step frames, save PNG, exit")
    args = ap.parse_args()

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    W, H = 1280, 720
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if args.shot else 0)
    pygame.display.set_mode((W, H), flags, vsync=0)
    aspect = W / H

    compute = compileProgram(compileShader(CS, GL_COMPUTE_SHADER))
    vao = glGenVertexArrays(1)  # core profile needs a VAO bound before validating the render program
    glBindVertexArray(vao)
    render = compileProgram(compileShader(VS, GL_VERTEX_SHADER), compileShader(FS, GL_FRAGMENT_SHADER))

    buf = glGenBuffers(1)
    def reset():
        data = seed(args.count, aspect)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_COPY)
    reset()
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf)

    # the same buffer doubles as the vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, buf)
    for loc, off in ((0, 0), (1, 16)):
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(off))

    U = lambda prog, name: glGetUniformLocation(prog, name)
    glUseProgram(compute)
    glUniform1ui(U(compute, "count"), args.count)
    glUniform1f(U(compute, "aspect"), aspect)
    glUseProgram(render)
    glUniform1f(U(render, "aspect"), aspect)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # additive glow
    glClearColor(0, 0, 0.02, 1)
    groups = (args.count + 255) // 256

    t, paused, frames, fps_t, last = 0.0, False, 0, time.perf_counter(), time.perf_counter()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                reset()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused

        now = time.perf_counter()
        dt = 1 / 60 if args.shot else min(now - last, 1 / 30)
        last = now

        lmb, _, rmb = pygame.mouse.get_pressed()
        if (lmb or rmb) and not args.shot:
            mx, my = pygame.mouse.get_pos()
            target, strength = ((mx / W * 2 - 1) * aspect, 1 - my / H * 2), (1.5 if lmb else -1.5)
        else:  # drifting Lissajous attractor
            target, strength = (0.6 * aspect * np.sin(t * 0.5), 0.5 * np.sin(t * 0.8)), 0.6

        if not paused:
            t += dt
            glUseProgram(compute)
            glUniform1f(U(compute, "dt"), dt)
            glUniform2f(U(compute, "attractor"), *target)
            glUniform1f(U(compute, "strength"), strength)
            glDispatchCompute(groups, 1, 1)
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(render)
        glDrawArrays(GL_POINTS, 0, args.count)

        frames += 1
        if args.shot and frames == 240:
            data = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
            surf = pygame.transform.flip(pygame.image.frombuffer(data, (W, H), "RGB"), False, True)
            pygame.image.save(surf, args.shot)
            print(f"saved {args.shot} ({args.count:,} particles)")
            break
        pygame.display.flip()
        if not args.shot and now - fps_t > 0.5:
            pygame.display.set_caption(f"PyOpenGL particles | {args.count:,} | {frames / (now - fps_t):.0f} fps")
            frames, fps_t = 0, now

    pygame.quit()


if __name__ == "__main__":
    main()
