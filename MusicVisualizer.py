#!/usr/bin/env python3
import sys, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import glfw
import moderngl

def load_audio(path):
    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        import librosa
        data, sr = librosa.load(path, sr=None, mono=True)
        return data.astype("float32"), sr

def profile_mood(audio, sr):
    energy = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    fft = np.fft.rfft(audio)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    brightness = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
    spec_mean = brightness
    complexity = float(np.sqrt(np.sum(((freqs - spec_mean) ** 2) * mag) / (np.sum(mag) + 1e-12)))
    dom_idx = int(np.argmax(mag))
    dom_freq = float(freqs[dom_idx])
    return {"energy": energy, "brightness": brightness, "complexity": complexity, "dominant_freq": dom_freq}

def mood_to_hue_params(m):
    brightness_norm = np.clip(m["brightness"] / 8000.0, 0.0, 1.0)
    dom_freq_norm = np.clip(m["dominant_freq"] / 5000.0, 0.0, 1.0)
    complexity_norm = np.clip(m["complexity"] / 4000.0, 0.0, 1.0)
    base_hue = (0.7 * dom_freq_norm + 0.3 * brightness_norm) % 1.0
    base_hue = (base_hue + 2.0/3.0) % 1.0
    hue_spread = 0.06 + 0.18 * complexity_norm
    return float(base_hue), float(hue_spread)

if len(sys.argv) < 2:
    print("usage: python viz_obs.py audiofile [delay_seconds]")
    sys.exit(1)

audio, sr = load_audio(sys.argv[1])
delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

# extend audio for pre/post visualization
pre_post_seconds = 1.0
pad_samples = int(pre_post_seconds * sr)
audio = np.pad(audio, (pad_samples, pad_samples))
total_samples = len(audio)

mood = profile_mood(audio, sr)
base_hue, hue_spread = mood_to_hue_params(mood)
print("MOOD:", mood, "base_hue:", base_hue, "hue_spread:", hue_spread)

if not glfw.init():
    raise RuntimeError("glfw init failed")

monitor = glfw.get_primary_monitor()
mode = glfw.get_video_mode(monitor)
W, H = mode.size.width, mode.size.height

# Make a windowed surface (not exclusive fullscreen). Position at 0,0 to cover screen.
glfw.window_hint(glfw.DECORATED, glfw.FALSE)
glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
window = glfw.create_window(W, H, "Visualizer (windowed)", None, None)
glfw.set_window_pos(window, 0, 0)
glfw.make_context_current(window)

# create moderngl context
ctx = moderngl.create_context()
ctx.enable(moderngl.BLEND)
ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

# create an offscreen framebuffer matching window size
scene_tex = ctx.texture((W, H), components=3)
scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
scene_fbo = ctx.framebuffer(color_attachments=[scene_tex])

# --- shaders ---
noise_prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec2 in_pos;
        out vec2 uv;
        void main() {
            uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330
        in vec2 uv;
        out vec4 frag;
        uniform float time;
        uniform float amp;
        uniform float freq;
        uniform float base_hue;
        uniform float hue_spread;

        float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
        float noise(vec2 p){
            vec2 i = floor(p);
            vec2 f = fract(p);
            float a = hash(i), b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0)), d = hash(i + vec2(1.0, 1.0));
            vec2 u = f*f*(3.0-2.0*f);
            return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
        }

        vec3 hsv2rgb(float h, float s, float v){
            float c = v * s;
            float hp = h * 6.0;
            float x = c * (1.0 - abs(mod(hp,2.0)-1.0));
            vec3 rgb = vec3(0.0);
            if (hp < 1.0) rgb = vec3(c,x,0.0);
            else if (hp < 2.0) rgb = vec3(x,c,0.0);
            else if (hp < 3.0) rgb = vec3(0.0,c,x);
            else if (hp < 4.0) rgb = vec3(0.0,x,c);
            else if (hp < 5.0) rgb = vec3(x,0.0,c);
            else rgb = vec3(c,0.0,x);
            return rgb + vec3(v - c);
        }

        void main() {
            vec2 p = (uv - 0.5) * 6.0;
            float n1 = noise(p + time * 0.45);
            float n2 = noise(p * 0.5 - time * 0.18);
            float n3 = noise(p * 1.5 + time * 0.35);
            float combined = pow(n1 + n2*0.6 + n3*0.3, max(0.0001, amp*4.0));
            float brightness = clamp(amp * 2.0 * combined, 0.0, 1.0);
            float hue = fract(base_hue + freq * 0.001 * hue_spread * 0.5);
            float sat = 0.6 + 0.4 * n2;
            float val = pow(brightness, max(0.0001, freq * 0.001));
            val = mix(val, 1.0, clamp(amp*0.5, 0.0, 1.0));
            vec3 rgb = hsv2rgb(hue, sat, val);
            frag = vec4(rgb, 1.0);
        }
    """
)

# simple white fill shader for the blobby waveform
wave_prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec2 in_pos;
        void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
    """,
    fragment_shader="""
        #version 330
        out vec4 color;
        void main(){ color = vec4(1.0, 1.0, 1.0, 0.95); }
    """
)

# shader to blit the scene texture to the default framebuffer (screen)
blit_prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec2 in_pos;
        out vec2 uv;
        void main(){ uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
    """,
    fragment_shader="""
        #version 330
        uniform sampler2D tex;
        in vec2 uv;
        out vec4 frag;
        void main(){ frag = texture(tex, uv); }
    """
)

# fullscreen quad
quad = np.array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1], dtype='f4')
vbo_quad = ctx.buffer(quad.tobytes())
vao_noise = ctx.simple_vertex_array(noise_prog, vbo_quad, 'in_pos')
vao_blit = ctx.simple_vertex_array(blit_prog, vbo_quad, 'in_pos')

# blobby waveform geometry (triangle strip)
window_samples = 1024
xs = np.linspace(-1.0, 1.0, window_samples, dtype='f4')
wave_vertices = np.zeros((window_samples*2, 2), dtype='f4')
wave_vertices[0::2, 0] = xs
wave_vertices[1::2, 0] = xs
vbo_wave = ctx.buffer(wave_vertices.tobytes())
vao_wave = ctx.simple_vertex_array(wave_prog, vbo_wave, 'in_pos')

# ready to play
if delay > 0:
    print(f"Waiting {delay:.2f}s before starting audio (use to start OBS)...")
    time.sleep(delay)

sd.play(audio, sr)
start = time.time()
smooth_amp = 0.0
smooth_freq = 0.0

# main loop: render into offscreen FBO, then blit to window framebuffer
while not glfw.window_should_close(window):
    glfw.poll_events()
    t = time.time() - start
    idx = int(t * sr)
    if idx >= total_samples:
        break

    block = audio[idx:idx + window_samples]
    if len(block) < window_samples:
        block = np.pad(block, (0, window_samples - len(block)))

    amp = np.sqrt(np.mean(block**2))
    smooth_amp = smooth_amp * 0.95 + amp * 0.05

    windowed = block * np.hanning(len(block))
    fft = np.fft.rfft(windowed)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(block), 1.0 / sr)
    mag_sum = np.sum(mag) + 1e-12
    dom_freq = float(np.sum(freqs * mag) / mag_sum)
    smooth_freq = smooth_freq * 0.98 + dom_freq * 0.02

    # update uniforms
    noise_prog['time'].value = float(t)
    noise_prog['amp'].value = float(np.clip(smooth_amp * 2.0, 0.0, 1.0))
    noise_prog['freq'].value = float(smooth_freq)
    noise_prog['base_hue'].value = base_hue
    noise_prog['hue_spread'].value = hue_spread

    # prepare blobby waveform vertices (top + center)
    y_scale = 0.5 * (0.25 + smooth_amp * 0.9)
    wave_vertices[0::2, 1] = (block.astype('f4')) * y_scale
    wave_vertices[1::2, 1] = 0.0
    vbo_wave.write(wave_vertices.tobytes())

    # Render into offscreen FBO (scene_fbo)
    scene_fbo.use()
    ctx.viewport = (0, 0, W, H)
    ctx.clear(0.0, 0.0, 0.0, 0.0)
    vao_noise.render()
    ctx.line_width = 2.0
    vao_wave.render(mode=moderngl.TRIANGLE_STRIP)

    # Blit the FBO texture to the default framebuffer (the window)
    ctx.screen.use()
    ctx.viewport = (0, 0, W, H)
    scene_tex.use(location=0)
    blit_prog['tex'].value = 0
    vao_blit.render()

    glfw.swap_buffers(window)

sd.stop()
glfw.terminate()

