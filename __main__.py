import os
import math
import time
import argparse
import xml.etree.ElementTree as ET
import taichi as ti
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageChops

ti.init(arch=ti.cuda)

# Rendering Constants
FINAL_WIDTH = 5120
FINAL_HEIGHT = 1440
OVERSAMPLE = 2             # Renders at 2x resolution (4K) then downscales for anti-aliasing
WIDTH = FINAL_WIDTH * OVERSAMPLE
HEIGHT = FINAL_HEIGHT * OVERSAMPLE
NUM_THREADS = 200_000      # Number of parallel GPU threads
ITERATIONS = 50_000        # INCREASED: 10 Billion total points for true smoothness
BURN_IN = 50               # Number of initial iterations to skip (letting the math settle)
GAMMA = 2.2                # Gamma correction for final image
BRIGHTNESS = 2.0           # Global brightness multiplier
VIBRANCE = 1.4             # Saturation boost for dense areas
BLOOM_INTENSITY = 0.4      # How much the bright areas glow (0.0 to 1.0)
NOISE_FALLOFF = 1.5        # NEW: Exponential curve to smoothly fade out stray grain

# Data structure mapping to hold our Affine Transforms and Variation weights in VRAM
# Taichi structs ensure memory alignment on the GPU
XFormStruct = ti.types.struct(
    a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32, e=ti.f32, f=ti.f32, # Affine matrix
    weight=ti.f32,     # Probability of selection
    color_idx=ti.f32,  # Target color coordinate (0.0 to 1.0)
    
    # Variations (The mathematical deformations)
    v_linear=ti.f32,
    v_sinusoidal=ti.f32,
    v_spherical=ti.f32,
    v_swirl=ti.f32,
    v_horseshoe=ti.f32,
    v_polar=ti.f32,
    v_handkerchief=ti.f32,
    v_heart=ti.f32,
    v_disc=ti.f32,
    v_spiral=ti.f32,
    v_hyperbolic=ti.f32,
    v_diamond=ti.f32,
    v_ex=ti.f32,
    v_julia=ti.f32,
    v_bent=ti.f32,
    v_waves=ti.f32,
    v_fisheye=ti.f32,
    v_popcorn=ti.f32,
    v_exponential=ti.f32,
    v_power=ti.f32,
    v_cosine=ti.f32,
    v_rings=ti.f32,
    v_fan=ti.f32,
    v_eyefish=ti.f32,
    v_bubble=ti.f32,
    v_cylinder=ti.f32,
    v_noise=ti.f32,
    v_blur=ti.f32,
    v_gaussian_blur=ti.f32
)

# Accumulator holds R, G, B, and Density (Hit Count) for every pixel
accumulator = ti.Vector.field(4, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# The rendered image buffer (normalized 0.0 to 1.0 RGB)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# Global variable to hold the maximum density found during tone-mapping
max_density = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def render_flame_kernel(
    num_xforms: ti.template(), 
    xforms: ti.template(), 
    palette: ti.template(), 
    cam_scale: ti.f32, 
    cam_x: ti.f32, 
    cam_y: ti.f32
):
    # This loop runs massively in parallel on your RTX 4090
    for thread_id in range(NUM_THREADS):
        # Initialize a random starting point and color coordinate
        x = ti.random(ti.f32) * 2.0 - 1.0
        y = ti.random(ti.f32) * 2.0 - 1.0
        c = ti.random(ti.f32) 
        
        # The Iterated Function System (IFS) loop
        for step in range(ITERATIONS):
            # 1. Roulette Wheel Selection: Pick a random transform based on weights
            rand_choice = ti.random(ti.f32)
            cumulative_weight = 0.0
            chosen_idx = 0
            
            for j in range(num_xforms):
                cumulative_weight += xforms[j].weight
                if rand_choice <= cumulative_weight:
                    chosen_idx = j
                    break 
            
            xf = xforms[chosen_idx]
            
            # 2. Apply Affine Transform (Standard Matrix Multiplication + Translation)
            nx = xf.a * x + xf.c * y + xf.e
            ny = xf.b * x + xf.d * y + xf.f
            
            # 3. Apply Non-Linear Variations (The "Apophysis" Magic)
            final_x = 0.0
            final_y = 0.0
            
            # Pre-calculate common math to save GPU cycles
            r2 = nx * nx + ny * ny
            r = ti.math.sqrt(r2)
            theta = ti.math.atan2(ny, nx)
            
            # Linear
            if xf.v_linear > 0.0:
                final_x += xf.v_linear * nx
                final_y += xf.v_linear * ny
                
            # Sinusoidal
            if xf.v_sinusoidal > 0.0:
                final_x += xf.v_sinusoidal * ti.math.sin(nx)
                final_y += xf.v_sinusoidal * ti.math.sin(ny)
                
            # Spherical
            if xf.v_spherical > 0.0:
                r2_safe = ti.math.max(r2, 1e-10)
                final_x += xf.v_spherical * (nx / r2_safe)
                final_y += xf.v_spherical * (ny / r2_safe)
                
            # Swirl
            if xf.v_swirl > 0.0:
                sin_r2 = ti.math.sin(r2)
                cos_r2 = ti.math.cos(r2)
                final_x += xf.v_swirl * (nx * sin_r2 - ny * cos_r2)
                final_y += xf.v_swirl * (nx * cos_r2 + ny * sin_r2)
                
            # Horseshoe
            if xf.v_horseshoe > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                final_x += xf.v_horseshoe * ((nx - ny) * (nx + ny)) / r_safe
                final_y += xf.v_horseshoe * (2.0 * nx * ny) / r_safe
                
            # Polar
            if xf.v_polar > 0.0:
                final_x += xf.v_polar * (theta / ti.math.pi)
                final_y += xf.v_polar * (r - 1.0)
                
            # Handkerchief
            if xf.v_handkerchief > 0.0:
                final_x += xf.v_handkerchief * r * ti.math.sin(theta + r)
                final_y += xf.v_handkerchief * r * ti.math.cos(theta - r)
                
            # Heart
            if xf.v_heart > 0.0:
                final_x += xf.v_heart * r * ti.math.sin(theta * r)
                final_y += xf.v_heart * -r * ti.math.cos(theta * r)

            # Disc
            if xf.v_disc > 0.0:
                final_x += xf.v_disc * (theta / ti.math.pi) * ti.math.sin(ti.math.pi * r)
                final_y += xf.v_disc * (theta / ti.math.pi) * ti.math.cos(ti.math.pi * r)

            # Spiral
            if xf.v_spiral > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                final_x += xf.v_spiral * (ti.math.cos(theta) + ti.math.sin(r)) / r_safe
                final_y += xf.v_spiral * (ti.math.sin(theta) - ti.math.cos(r)) / r_safe
                
            # Hyperbolic
            if xf.v_hyperbolic > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                final_x += xf.v_hyperbolic * ti.math.sin(theta) / r_safe
                final_y += xf.v_hyperbolic * r * ti.math.cos(theta)

            # Diamond
            if xf.v_diamond > 0.0:
                final_x += xf.v_diamond * ti.math.sin(theta) * ti.math.cos(r)
                final_y += xf.v_diamond * ti.math.cos(theta) * ti.math.sin(r)
                
            # Ex
            if xf.v_ex > 0.0:
                n0 = ti.math.sin(theta + r)
                n1 = ti.math.cos(theta - r)
                m0 = n0 * n0 * n0
                m1 = n1 * n1 * n1
                final_x += xf.v_ex * r * (m0 + m1)
                final_y += xf.v_ex * r * (m0 - m1)
                
            # Julia
            if xf.v_julia > 0.0:
                r_julia = ti.math.sqrt(r)
                theta_julia = theta * 0.5 + ti.math.pi * (ti.random(ti.i32) % 2)
                final_x += xf.v_julia * r_julia * ti.math.cos(theta_julia)
                final_y += xf.v_julia * r_julia * ti.math.sin(theta_julia)
                
            # Bent
            if xf.v_bent > 0.0:
                bent_x = nx
                bent_y = ny
                if nx >= 0.0 and ny >= 0.0:
                    pass
                elif nx < 0.0 and ny >= 0.0:
                    bent_x = 2.0 * nx
                elif nx >= 0.0 and ny < 0.0:
                    bent_y = ny * 0.5
                else:
                    bent_x = 2.0 * nx
                    bent_y = ny * 0.5
                final_x += xf.v_bent * bent_x
                final_y += xf.v_bent * bent_y

            # Waves
            if xf.v_waves > 0.0:
                # Standard parameters for waves
                dx = nx + xf.b * ti.math.sin(ny / (xf.c * xf.c + 1e-10))
                dy = ny + xf.e * ti.math.sin(nx / (xf.f * xf.f + 1e-10))
                final_x += xf.v_waves * dx
                final_y += xf.v_waves * dy
                
            # Fisheye
            if xf.v_fisheye > 0.0:
                r_fisheye = 2.0 / (r + 1.0)
                final_x += xf.v_fisheye * r_fisheye * ny
                final_y += xf.v_fisheye * r_fisheye * nx

            # Popcorn
            if xf.v_popcorn > 0.0:
                dx = nx + xf.c * ti.math.sin(ti.math.tan(3.0 * ny))
                dy = ny + xf.f * ti.math.sin(ti.math.tan(3.0 * nx))
                final_x += xf.v_popcorn * dx
                final_y += xf.v_popcorn * dy

            # Exponential
            if xf.v_exponential > 0.0:
                exp_nx = ti.math.exp(nx - 1.0)
                final_x += xf.v_exponential * exp_nx * ti.math.cos(ti.math.pi * ny)
                final_y += xf.v_exponential * exp_nx * ti.math.sin(ti.math.pi * ny)

            # Power
            if xf.v_power > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                pow_theta = ti.math.pow(r_safe, ti.math.sin(theta))
                final_x += xf.v_power * pow_theta * ti.math.cos(theta)
                final_y += xf.v_power * pow_theta * ti.math.sin(theta)

            # Cosine
            if xf.v_cosine > 0.0:
                exp_ny = ti.math.exp(ny)
                exp_neg_ny = ti.math.exp(-ny)
                cosh_ny = 0.5 * (exp_ny + exp_neg_ny)
                sinh_ny = 0.5 * (exp_ny - exp_neg_ny)
                final_x += xf.v_cosine * ti.math.cos(ti.math.pi * nx) * cosh_ny
                final_y += xf.v_cosine * -ti.math.sin(ti.math.pi * nx) * sinh_ny

            # Rings
            if xf.v_rings > 0.0:
                dx = xf.c * xf.c + 1e-10
                r_rings = ((r + dx) % (2.0 * dx)) - dx + r * (1.0 - dx)
                final_x += xf.v_rings * r_rings * ti.math.cos(theta)
                final_y += xf.v_rings * r_rings * ti.math.sin(theta)

            # Fan
            if xf.v_fan > 0.0:
                dx = ti.math.pi * (xf.c * xf.c + 1e-10)
                dx2 = dx * 0.5
                t = theta + xf.f - ti.math.floor((theta + xf.f) / dx) * dx
                if t > dx2:
                    final_x += xf.v_fan * r * ti.math.cos(theta - dx2)
                    final_y += xf.v_fan * r * ti.math.sin(theta - dx2)
                else:
                    final_x += xf.v_fan * r * ti.math.cos(theta + dx2)
                    final_y += xf.v_fan * r * ti.math.sin(theta + dx2)

            # Eyefish
            if xf.v_eyefish > 0.0:
                r_eyefish = 2.0 / (r + 1.0)
                final_x += xf.v_eyefish * r_eyefish * nx
                final_y += xf.v_eyefish * r_eyefish * ny

            # Bubble
            if xf.v_bubble > 0.0:
                r_bubble = 4.0 / (r2 + 4.0)
                final_x += xf.v_bubble * r_bubble * nx
                final_y += xf.v_bubble * r_bubble * ny

            # Cylinder
            if xf.v_cylinder > 0.0:
                final_x += xf.v_cylinder * ti.math.sin(nx)
                final_y += xf.v_cylinder * ny

            # Noise
            if xf.v_noise > 0.0:
                rx = ti.random(ti.f32)
                ry = ti.random(ti.f32) * 2.0 * ti.math.pi
                final_x += xf.v_noise * rx * nx * ti.math.cos(ry)
                final_y += xf.v_noise * rx * ny * ti.math.sin(ry)

            # Blur
            if xf.v_blur > 0.0:
                blur_r = ti.random(ti.f32)
                blur_theta = ti.random(ti.f32) * 2.0 * ti.math.pi
                final_x += xf.v_blur * blur_r * ti.math.cos(blur_theta)
                final_y += xf.v_blur * blur_r * ti.math.sin(blur_theta)

            # Gaussian Blur
            if xf.v_gaussian_blur > 0.0:
                # Box-Muller transform for normal distribution
                u1 = ti.math.max(ti.random(ti.f32), 1e-10)
                u2 = ti.random(ti.f32)
                z0 = ti.math.sqrt(-2.0 * ti.math.log(u1)) * ti.math.cos(2.0 * ti.math.pi * u2)
                z1 = ti.math.sqrt(-2.0 * ti.math.log(u1)) * ti.math.sin(2.0 * ti.math.pi * u2)
                final_x += xf.v_gaussian_blur * z0
                final_y += xf.v_gaussian_blur * z1

            # Update coordinates for the next iteration
            x = final_x
            y = final_y
            
            # 4. Color Calculation (Blend current color with transform color)
            c = (c + xf.color_idx) * 0.5
            
            # 5. Plot the point (if we have passed the burn-in phase)
            if step > BURN_IN:
                # Map mathematical space to screen space
                px_float = (x - cam_x) * cam_scale + (WIDTH / 2.0)
                py_float = (y - cam_y) * cam_scale + (HEIGHT / 2.0)
                
                px = ti.cast(px_float, ti.i32)
                py = ti.cast(py_float, ti.i32)
                
                # If point is on screen, map color from palette and accumulate
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # Look up color in 256-color palette
                    pal_idx = ti.cast(c * 255.0, ti.i32)
                    pal_idx = ti.math.clamp(pal_idx, 0, 255)
                    color = palette[pal_idx]
                    
                    # ATOMIC ADDITION: Safely update VRAM even with 100k threads hitting it
                    # Accumulator vector is [R, G, B, Density]
                    ti.atomic_add(accumulator[px, py][0], color[0])
                    ti.atomic_add(accumulator[px, py][1], color[1])
                    ti.atomic_add(accumulator[px, py][2], color[2])
                    ti.atomic_add(accumulator[px, py][3], 1.0) # Increment hit count

@ti.kernel
def find_max_density_kernel():
    # Reset max density
    max_density[None] = 0.0
    # Search all pixels for the highest hit count
    for i, j in accumulator:
        ti.atomic_max(max_density[None], accumulator[i, j][3])

@ti.kernel
def apply_tone_mapping_kernel():
    # Map the linear accumulation to logarithmic visual space
    max_d = max_density[None]
    log_max = ti.math.log(max_d)
    
    for i, j in accumulator:
        dens = accumulator[i, j][3]
        # Ignore empty space and single rogue hits
        if dens > 1.0:
            # 1. Calculate Logarithmic alpha/exposure
            alpha = ti.math.log(dens) / log_max
            
            # NEW: Soft Noise Gate. By raising the fraction to a power, 
            # low-density grain drops to black smoothly without a harsh, jagged edge.
            alpha = ti.math.pow(alpha, NOISE_FALLOFF)
            
            # 2. Extract base color averages
            r = accumulator[i, j][0] / dens
            g = accumulator[i, j][1] / dens
            b = accumulator[i, j][2] / dens
            
            # 3. HDR Vibrance (Boost saturation before gamma clamping)
            # Calculate perceived luminance (brightness) of the color
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            
            # Push the colors away from grayscale to increase purity
            r = luminance + (r - luminance) * VIBRANCE
            g = luminance + (g - luminance) * VIBRANCE
            b = luminance + (b - luminance) * VIBRANCE
            
            # 4. Apply Brightness, Alpha, and Gamma Correction
            # ti.math.max is used to prevent negative bases in the pow() function from the vibrance math
            final_r = ti.math.pow(ti.math.max(r * alpha * BRIGHTNESS, 0.0), 1.0 / GAMMA)
            final_g = ti.math.pow(ti.math.max(g * alpha * BRIGHTNESS, 0.0), 1.0 / GAMMA)
            final_b = ti.math.pow(ti.math.max(b * alpha * BRIGHTNESS, 0.0), 1.0 / GAMMA)
            
            # Clamp to safe 0-1 range
            pixels[i, j] = ti.math.clamp(ti.math.vec3(final_r, final_g, final_b), 0.0, 1.0)
        else:
            pixels[i, j] = ti.math.vec3(0.0, 0.0, 0.0)

class ApophysisRenderer:
    def __init__(self, flame_path, zoom_multiplier=1.0):
        self.flame_path = flame_path
        self.zoom_multiplier = zoom_multiplier
        self.xforms = []
        self.palette_data = []
        self.camera = {"scale": 100.0, "x": 0.0, "y": 0.0}
        
        # Supported variations list (Updated with 7X additions)
        self.supported_vars = [
            "linear", "sinusoidal", "spherical", "swirl", "horseshoe", 
            "polar", "handkerchief", "heart", "disc", "spiral", 
            "hyperbolic", "diamond", "ex", "julia", "bent", "waves", 
            "fisheye", "popcorn", "exponential", "power", "cosine", 
            "rings", "fan", "eyefish", "bubble", "cylinder", "noise", 
            "blur", "gaussian_blur"
        ]
        
    def parse_flame(self):
        print(f"Parsing {self.flame_path}...")
        tree = ET.parse(self.flame_path)
        root = tree.getroot()
        flame = root.find('flame')
        
        if flame is None:
            # Handle case where root IS the flame tag
            flame = root if root.tag == 'flame' else None
            
        if flame is None:
            raise ValueError("Invalid .flame file format.")

        # 1. Parse Camera Settings
        if 'scale' in flame.attrib:
            # We multiply by OVERSAMPLE to ensure the fractal size stays consistent 
            # regardless of internal render resolution, and then apply user zoom
            self.camera['scale'] = float(flame.attrib['scale']) * OVERSAMPLE * self.zoom_multiplier
        if 'center' in flame.attrib:
            cx, cy = map(float, flame.attrib['center'].split())
            self.camera['x'] = cx
            self.camera['y'] = cy

        # 2. Parse Palette
        palette_tag = flame.find('palette')
        if palette_tag is not None and palette_tag.text:
            # Apophysis palette is usually a long string of hex codes
            text = palette_tag.text.replace("\n", "").replace(" ", "").strip()
            # If it's a solid block of hex codes (RRGGBB)
            for i in range(256):
                if i*6+6 <= len(text):
                    hex_str = text[i*6 : i*6+6]
                    r = int(hex_str[0:2], 16) / 255.0
                    g = int(hex_str[2:4], 16) / 255.0
                    b = int(hex_str[4:6], 16) / 255.0
                    self.palette_data.append((r, g, b))
        
        # Fallback palette if missing
        if len(self.palette_data) < 256:
            print("Warning: valid palette not found. Generating default heat-map palette.")
            self.palette_data = [(i/255.0, (i/255.0)**2, 0.0) for i in range(256)]

        # 3. Parse XForms (Transforms)
        total_weight = 0.0
        for xform_tag in flame.findall('xform'):
            # Coefficients: a b c d e f
            coef_str = xform_tag.attrib.get('coefs', '1 0 0 1 0 0')
            coefs = list(map(float, coef_str.split()))
            
            weight = float(xform_tag.attrib.get('weight', 1.0))
            color_idx = float(xform_tag.attrib.get('color', 0.0))
            
            total_weight += weight
            
            xf_data = {
                'coefs': coefs,
                'weight': weight,
                'color_idx': color_idx
            }
            
            # Extract variations dynamically
            for v in self.supported_vars:
                xf_data[f'v_{v}'] = float(xform_tag.attrib.get(v, 0.0))
                
            self.xforms.append(xf_data)
            
        # Normalize weights so they add up to 1.0 (for the GPU probability logic)
        for xf in self.xforms:
            xf['weight'] /= total_weight
            
        print(f"Parsed {len(self.xforms)} transforms successfully.")

    def render(self, output_path="output.png"):
        self.parse_flame()
        
        # 1. Allocate and fill GPU Fields
        num_xfs = len(self.xforms)
        d_xforms = XFormStruct.field(shape=(num_xfs,))
        d_palette = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        
        for i, xf in enumerate(self.xforms):
            d_xforms[i].a, d_xforms[i].b, d_xforms[i].c, d_xforms[i].d, d_xforms[i].e, d_xforms[i].f = xf['coefs']
            d_xforms[i].weight = xf['weight']
            d_xforms[i].color_idx = xf['color_idx']
            
            for v in self.supported_vars:
                setattr(d_xforms[i], f'v_{v}', xf[f'v_{v}'])
                
        for i in range(256):
            d_palette[i] = ti.Vector(self.palette_data[i])
            
        # Clear accumulators
        accumulator.fill(0)
        
        # 2. Execute Chaos Game Kernel
        print(f"Executing Compute Shader: {NUM_THREADS} threads * {ITERATIONS} iterations...")
        start_time = time.time()
        
        render_flame_kernel(
            num_xforms=num_xfs, 
            xforms=d_xforms, 
            palette=d_palette, 
            cam_scale=self.camera['scale'], 
            cam_x=self.camera['x'], 
            cam_y=self.camera['y']
        )
        
        # Ensure GPU operations are complete
        ti.sync()
        print(f"Compute finished in {time.time() - start_time:.2f} seconds.")
        
        # 3. Tone Mapping
        print("Applying Log-Density Tone Mapping...")
        find_max_density_kernel()
        apply_tone_mapping_kernel()
        ti.sync()
        
        # 4. Convert GPU field to Numpy array and save via PIL
        # Taichi pixel format is (WIDTH, HEIGHT, 3), PIL expects (HEIGHT, WIDTH, 3)
        img_np = pixels.to_numpy()
        img_np = np.swapaxes(img_np, 0, 1) # Flip axes
        img_np = img_np[::-1, :, :]        # Flip Y axis (Apophysis/Math coordinates are bottom-up)
        
        img_uint8 = (img_np * 255.0).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'RGB')
        
        # Apply Supersampling Anti-Aliasing (SSAA) by downscaling
        if OVERSAMPLE > 1:
            print(f"Downscaling from {WIDTH}x{HEIGHT} to {FINAL_WIDTH}x{FINAL_HEIGHT} for Anti-Aliasing...")
            img = img.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS)
            
        # --- Cinematic Post-Processing Pipeline ---
        print("Applying Cinematic Post-Processing (Bloom & Color Grade)...")
        
        # 1. Optical Bloom (Glow)
        # Blur a copy of the image based on screen width (1.5% radius)
        blur_radius = FINAL_WIDTH * 0.015
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # Screen blend the glowing blur over the original image
        screened_img = ImageChops.screen(img, blurred_img)
        # Blend original with screened version based on user intensity
        img = Image.blend(img, screened_img, BLOOM_INTENSITY)
        
        # 2. Final Color Grading (Contrast and Saturation Pop)
        img = ImageEnhance.Contrast(img).enhance(1.15) # Darken shadows slightly
        img = ImageEnhance.Color(img).enhance(1.1)   # Give final RGB values a slight punch
        
        img.save(output_path)
        print(f"Success! Fractal saved to {output_path}")

def generate_sample_flame(filepath):
    """Generates a beautiful Apophysis XML file combining Swirl and Spherical math."""
    xml_content = """<flames>
<flame name="Cosmic_Flower" version="Apophysis 7x" size="1920 1080" center="0 0" scale="250">
    <xform weight="0.5" color="0.0" swirl="0.8" linear="0.2" coefs="0.8 0.3 -0.3 0.8 0 0" />
    <xform weight="0.5" color="0.5" spherical="1.0" coefs="0.5 0 0 0.5 1.0 0.5" />
    <xform weight="0.5" color="1.0" horseshoe="0.8" coefs="0.3 0.5 -0.5 0.3 -1.0 -0.5" />
    <palette count="256">
"""
    # Generate a beautiful cyan to purple to orange gradient for the palette
    for i in range(256):
        r = int((math.sin(i * 0.05) * 0.5 + 0.5) * 255)
        g = int((math.sin(i * 0.05 + 2) * 0.5 + 0.5) * 255)
        b = int((math.sin(i * 0.05 + 4) * 0.5 + 0.5) * 255)
        xml_content += f"{r:02x}{g:02x}{b:02x}\n"
        
    xml_content += """    </palette>
</flame>
</flames>"""

    with open(filepath, "w") as f:
        f.write(xml_content)
    print(f"Generated sample flame file at {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an Apophysis .flame file using Taichi.")
    parser.add_argument("-i", "--input", type=str, default="sample_fractal.flame", help="Path to the input .flame file")
    parser.add_argument("-o", "--output", type=str, default="render_output.png", help="Path to the saved output image")
    parser.add_argument("-z", "--zoom", type=float, default=1.0, help="Zoom multiplier (e.g. 0.5 to zoom out, 2.0 to zoom in)")
    args = parser.parse_args()
    
    flame_file = args.input
    output_file = args.output
    zoom_factor = args.zoom
    
    # If using the default and it doesn't exist, generate the sample
    if not os.path.exists(flame_file) and flame_file == "sample_fractal.flame":
        generate_sample_flame(flame_file)
    elif not os.path.exists(flame_file):
        print(f"Error: The file '{flame_file}' does not exist.")
        exit(1)
        
    try:
        renderer = ApophysisRenderer(flame_file, zoom_multiplier=zoom_factor)
        renderer.render(output_file)
    except Exception as e:
        print(f"Error rendering fractal: {e}")
