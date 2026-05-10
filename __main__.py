import os
import math
import time
import xml.etree.ElementTree as ET
import taichi as ti
import numpy as np
from PIL import Image

# Initialize Taichi to use the highest performance GPU available (CUDA for RTX 4090)
ti.init(arch=ti.gpu)

# Rendering Constants
WIDTH = 1920
HEIGHT = 1080
NUM_THREADS = 200_000      # Number of parallel GPU threads
ITERATIONS = 5_000         # Number of chaos game iterations per thread (1 Billion total points)
BURN_IN = 50               # Number of initial iterations to skip (letting the math settle)
GAMMA = 2.2                # Gamma correction for final image
BRIGHTNESS = 2.0           # Global brightness multiplier

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
    v_diamond=ti.f32
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
        if dens > 0.0:
            # 1. Calculate Logarithmic alpha/exposure
            alpha = ti.math.log(dens) / log_max
            
            # 2. Extract base color averages
            r = accumulator[i, j][0] / dens
            g = accumulator[i, j][1] / dens
            b = accumulator[i, j][2] / dens
            
            # 3. Apply Brightness, Alpha, and Gamma Correction
            final_r = ti.math.pow(r * alpha * BRIGHTNESS, 1.0 / GAMMA)
            final_g = ti.math.pow(g * alpha * BRIGHTNESS, 1.0 / GAMMA)
            final_b = ti.math.pow(b * alpha * BRIGHTNESS, 1.0 / GAMMA)
            
            # Clamp to safe 0-1 range
            pixels[i, j] = ti.math.clamp(ti.math.vec3(final_r, final_g, final_b), 0.0, 1.0)
        else:
            pixels[i, j] = ti.math.vec3(0.0, 0.0, 0.0)

class ApophysisRenderer:
    def __init__(self, flame_path):
        self.flame_path = flame_path
        self.xforms = []
        self.palette_data = []
        self.camera = {"scale": 100.0, "x": 0.0, "y": 0.0}
        
        # Supported variations list
        self.supported_vars = [
            "linear", "sinusoidal", "spherical", "swirl", "horseshoe", 
            "polar", "handkerchief", "heart", "disc", "spiral", 
            "hyperbolic", "diamond"
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
            self.camera['scale'] = float(flame.attrib['scale'])
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
    flame_file = "sample_fractal.flame"
    output_file = "render_output.png"
    
    # If the user hasn't provided a flame file, build a gorgeous one for them to test
    if not os.path.exists(flame_file):
        generate_sample_flame(flame_file)
        
    try:
        renderer = ApophysisRenderer(flame_file)
        renderer.render(output_file)
    except Exception as e:
        print(f"Error rendering fractal: {e}")
