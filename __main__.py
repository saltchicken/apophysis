import os
import math
import time
import argparse
import xml.etree.ElementTree as ET
import taichi as ti
import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image, ImageFilter, ImageEnhance, ImageChops

# Initialize Taichi
ti.init(arch=ti.cuda)

# All supported non-linear variations
SUPPORTED_VARS = [
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", 
    "polar", "handkerchief", "heart", "disc", "spiral", 
    "hyperbolic", "diamond", "ex", "julia", "bent", "waves", 
    "fisheye", "popcorn", "exponential", "power", "cosine", 
    "rings", "fan", "eyefish", "bubble", "cylinder", "noise", 
    "blur", "gaussian_blur"
]

# Default rendering configuration. 
# GUI uses a fast version of this, CLI uses a high-quality version.
DEFAULT_CONFIG = {
    "FINAL_WIDTH": 800,
    "FINAL_HEIGHT": 600,
    "OVERSAMPLE": 1,
    "NUM_THREADS": 100_000,
    "ITERATIONS": 500,
    "BURN_IN": 20,
    "GAMMA": 2.2,
    "BRIGHTNESS": 2.0,
    "VIBRANCE": 1.4,
    "BLOOM_INTENSITY": 0.4,
    "MIN_DENSITY": 1.0, 
    "NOISE_FALLOFF": 1.5,
    "FILTER_RADIUS": 1,
    "FILTER_WEIGHT_CENTER": 1.0,
    "FILTER_WEIGHT_EDGE": 0.5,
}

# Taichi struct for holding affine and variation weights in VRAM
XFormStruct = ti.types.struct(
    a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32, e=ti.f32, f=ti.f32,
    weight=ti.f32, color_idx=ti.f32,
    v_linear=ti.f32, v_sinusoidal=ti.f32, v_spherical=ti.f32, v_swirl=ti.f32,
    v_horseshoe=ti.f32, v_polar=ti.f32, v_handkerchief=ti.f32, v_heart=ti.f32,
    v_disc=ti.f32, v_spiral=ti.f32, v_hyperbolic=ti.f32, v_diamond=ti.f32,
    v_ex=ti.f32, v_julia=ti.f32, v_bent=ti.f32, v_waves=ti.f32,
    v_fisheye=ti.f32, v_popcorn=ti.f32, v_exponential=ti.f32, v_power=ti.f32,
    v_cosine=ti.f32, v_rings=ti.f32, v_fan=ti.f32, v_eyefish=ti.f32,
    v_bubble=ti.f32, v_cylinder=ti.f32, v_noise=ti.f32, v_blur=ti.f32,
    v_gaussian_blur=ti.f32
)


# =====================================================================
# CORE TAICHI KERNELS (Decoupled from global state)
# =====================================================================

@ti.kernel
def render_flame_kernel(
    num_xforms: ti.template(), xforms: ti.template(), palette: ti.template(), 
    accumulator: ti.template(),
    cam_scale: ti.f32, cam_x: ti.f32, cam_y: ti.f32,
    width: ti.i32, height: ti.i32, iterations: ti.i32, num_threads: ti.i32, burn_in: ti.i32,
    filter_radius: ti.i32, weight_center: ti.f32, weight_edge: ti.f32
):
    for thread_id in range(num_threads):
        x = ti.random(ti.f32) * 2.0 - 1.0
        y = ti.random(ti.f32) * 2.0 - 1.0
        c = ti.random(ti.f32) 
        
        for step in range(iterations):
            rand_choice = ti.random(ti.f32)
            cumulative_weight = 0.0
            chosen_idx = 0
            
            for j in range(num_xforms):
                cumulative_weight += xforms[j].weight
                if rand_choice <= cumulative_weight:
                    chosen_idx = j
                    break 
            
            xf = xforms[chosen_idx]
            
            nx = xf.a * x + xf.c * y + xf.e
            ny = xf.b * x + xf.d * y + xf.f
            
            final_x = 0.0
            final_y = 0.0
            
            r2 = nx * nx + ny * ny
            r = ti.math.sqrt(r2)
            theta = ti.math.atan2(ny, nx)
            
            # [Variations Mathematics - Omitted for brevity but functional]
            if xf.v_linear > 0.0: final_x += xf.v_linear * nx; final_y += xf.v_linear * ny
            if xf.v_sinusoidal > 0.0: final_x += xf.v_sinusoidal * ti.math.sin(nx); final_y += xf.v_sinusoidal * ti.math.sin(ny)
            if xf.v_spherical > 0.0:
                r2_safe = ti.math.max(r2, 1e-10)
                final_x += xf.v_spherical * (nx / r2_safe); final_y += xf.v_spherical * (ny / r2_safe)
            if xf.v_swirl > 0.0:
                sin_r2 = ti.math.sin(r2); cos_r2 = ti.math.cos(r2)
                final_x += xf.v_swirl * (nx * sin_r2 - ny * cos_r2); final_y += xf.v_swirl * (nx * cos_r2 + ny * sin_r2)
            if xf.v_horseshoe > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                final_x += xf.v_horseshoe * ((nx - ny) * (nx + ny)) / r_safe; final_y += xf.v_horseshoe * (2.0 * nx * ny) / r_safe
            if xf.v_polar > 0.0: final_x += xf.v_polar * (theta / ti.math.pi); final_y += xf.v_polar * (r - 1.0)
            if xf.v_handkerchief > 0.0: final_x += xf.v_handkerchief * r * ti.math.sin(theta + r); final_y += xf.v_handkerchief * r * ti.math.cos(theta - r)
            if xf.v_heart > 0.0: final_x += xf.v_heart * r * ti.math.sin(theta * r); final_y += xf.v_heart * -r * ti.math.cos(theta * r)
            if xf.v_disc > 0.0: final_x += xf.v_disc * (theta / ti.math.pi) * ti.math.sin(ti.math.pi * r); final_y += xf.v_disc * (theta / ti.math.pi) * ti.math.cos(ti.math.pi * r)
            if xf.v_spiral > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                final_x += xf.v_spiral * (ti.math.cos(theta) + ti.math.sin(r)) / r_safe; final_y += xf.v_spiral * (ti.math.sin(theta) - ti.math.cos(r)) / r_safe
            if xf.v_hyperbolic > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                final_x += xf.v_hyperbolic * ti.math.sin(theta) / r_safe; final_y += xf.v_hyperbolic * r * ti.math.cos(theta)
            if xf.v_diamond > 0.0: final_x += xf.v_diamond * ti.math.sin(theta) * ti.math.cos(r); final_y += xf.v_diamond * ti.math.cos(theta) * ti.math.sin(r)
            if xf.v_ex > 0.0:
                n0 = ti.math.sin(theta + r); n1 = ti.math.cos(theta - r)
                m0 = n0 * n0 * n0; m1 = n1 * n1 * n1
                final_x += xf.v_ex * r * (m0 + m1); final_y += xf.v_ex * r * (m0 - m1)
            if xf.v_julia > 0.0:
                r_julia = ti.math.sqrt(r)
                theta_julia = theta * 0.5 + ti.math.pi * (ti.random(ti.i32) % 2)
                final_x += xf.v_julia * r_julia * ti.math.cos(theta_julia); final_y += xf.v_julia * r_julia * ti.math.sin(theta_julia)
            if xf.v_bent > 0.0:
                bent_x = nx; bent_y = ny
                if nx >= 0.0 and ny >= 0.0: pass
                elif nx < 0.0 and ny >= 0.0: bent_x = 2.0 * nx
                elif nx >= 0.0 and ny < 0.0: bent_y = ny * 0.5
                else: bent_x = 2.0 * nx; bent_y = ny * 0.5
                final_x += xf.v_bent * bent_x; final_y += xf.v_bent * bent_y
            if xf.v_waves > 0.0:
                dx = nx + xf.b * ti.math.sin(ny / (xf.c * xf.c + 1e-10))
                dy = ny + xf.e * ti.math.sin(nx / (xf.f * xf.f + 1e-10))
                final_x += xf.v_waves * dx; final_y += xf.v_waves * dy
            if xf.v_fisheye > 0.0:
                r_fisheye = 2.0 / (r + 1.0)
                final_x += xf.v_fisheye * r_fisheye * ny; final_y += xf.v_fisheye * r_fisheye * nx
            if xf.v_popcorn > 0.0:
                dx = nx + xf.c * ti.math.sin(ti.math.tan(3.0 * ny))
                dy = ny + xf.f * ti.math.sin(ti.math.tan(3.0 * nx))
                final_x += xf.v_popcorn * dx; final_y += xf.v_popcorn * dy
            if xf.v_exponential > 0.0:
                exp_nx = ti.math.exp(nx - 1.0)
                final_x += xf.v_exponential * exp_nx * ti.math.cos(ti.math.pi * ny); final_y += xf.v_exponential * exp_nx * ti.math.sin(ti.math.pi * ny)
            if xf.v_power > 0.0:
                r_safe = ti.math.max(r, 1e-10)
                pow_theta = ti.math.pow(r_safe, ti.math.sin(theta))
                final_x += xf.v_power * pow_theta * ti.math.cos(theta); final_y += xf.v_power * pow_theta * ti.math.sin(theta)
            if xf.v_cosine > 0.0:
                exp_ny = ti.math.exp(ny); exp_neg_ny = ti.math.exp(-ny)
                cosh_ny = 0.5 * (exp_ny + exp_neg_ny); sinh_ny = 0.5 * (exp_ny - exp_neg_ny)
                final_x += xf.v_cosine * ti.math.cos(ti.math.pi * nx) * cosh_ny; final_y += xf.v_cosine * -ti.math.sin(ti.math.pi * nx) * sinh_ny
            if xf.v_rings > 0.0:
                dx = xf.c * xf.c + 1e-10
                r_rings = ((r + dx) % (2.0 * dx)) - dx + r * (1.0 - dx)
                final_x += xf.v_rings * r_rings * ti.math.cos(theta); final_y += xf.v_rings * r_rings * ti.math.sin(theta)
            if xf.v_fan > 0.0:
                dx = ti.math.pi * (xf.c * xf.c + 1e-10)
                dx2 = dx * 0.5
                t = theta + xf.f - ti.math.floor((theta + xf.f) / dx) * dx
                if t > dx2: final_x += xf.v_fan * r * ti.math.cos(theta - dx2); final_y += xf.v_fan * r * ti.math.sin(theta - dx2)
                else: final_x += xf.v_fan * r * ti.math.cos(theta + dx2); final_y += xf.v_fan * r * ti.math.sin(theta + dx2)
            if xf.v_eyefish > 0.0:
                r_eyefish = 2.0 / (r + 1.0)
                final_x += xf.v_eyefish * r_eyefish * nx; final_y += xf.v_eyefish * r_eyefish * ny
            if xf.v_bubble > 0.0:
                r_bubble = 4.0 / (r2 + 4.0)
                final_x += xf.v_bubble * r_bubble * nx; final_y += xf.v_bubble * r_bubble * ny
            if xf.v_cylinder > 0.0: final_x += xf.v_cylinder * ti.math.sin(nx); final_y += xf.v_cylinder * ny
            if xf.v_noise > 0.0:
                rx = ti.random(ti.f32); ry = ti.random(ti.f32) * 2.0 * ti.math.pi
                final_x += xf.v_noise * rx * nx * ti.math.cos(ry); final_y += xf.v_noise * rx * ny * ti.math.sin(ry)
            if xf.v_blur > 0.0:
                blur_r = ti.random(ti.f32); blur_theta = ti.random(ti.f32) * 2.0 * ti.math.pi
                final_x += xf.v_blur * blur_r * ti.math.cos(blur_theta); final_y += xf.v_blur * blur_r * ti.math.sin(blur_theta)
            if xf.v_gaussian_blur > 0.0:
                u1 = ti.math.max(ti.random(ti.f32), 1e-10); u2 = ti.random(ti.f32)
                z0 = ti.math.sqrt(-2.0 * ti.math.log(u1)) * ti.math.cos(2.0 * ti.math.pi * u2)
                z1 = ti.math.sqrt(-2.0 * ti.math.log(u1)) * ti.math.sin(2.0 * ti.math.pi * u2)
                final_x += xf.v_gaussian_blur * z0; final_y += xf.v_gaussian_blur * z1

            x = final_x
            y = final_y
            c = (c + xf.color_idx) * 0.5
            
            # Map point to screen
            if step > burn_in:
                px_float = (x - cam_x) * cam_scale + (width / 2.0)
                py_float = (y - cam_y) * cam_scale + (height / 2.0)
                
                px = ti.cast(px_float, ti.i32)
                py = ti.cast(py_float, ti.i32)
                
                if -filter_radius <= px < width + filter_radius and -filter_radius <= py < height + filter_radius:
                    pal_idx = ti.cast(c * 255.0, ti.i32)
                    pal_idx = ti.math.clamp(pal_idx, 0, 255)
                    color = palette[pal_idx]
                    
                    for dx in range(-filter_radius, filter_radius + 1):
                        for dy in range(-filter_radius, filter_radius + 1):
                            splat_x = px + dx
                            splat_y = py + dy
                            
                            if 0 <= splat_x < width and 0 <= splat_y < height:
                                dist2 = dx*dx + dy*dy
                                weight = 0.0
                                if dist2 == 0: weight = weight_center
                                elif dist2 <= filter_radius * filter_radius: weight = weight_edge / ti.math.sqrt(float(dist2))
                                
                                if weight > 0.0:
                                    ti.atomic_add(accumulator[splat_x, splat_y][0], color[0] * weight)
                                    ti.atomic_add(accumulator[splat_x, splat_y][1], color[1] * weight)
                                    ti.atomic_add(accumulator[splat_x, splat_y][2], color[2] * weight)
                                    ti.atomic_add(accumulator[splat_x, splat_y][3], weight)

@ti.kernel
def find_max_density_kernel(accumulator: ti.template(), max_density: ti.template()):
    max_density[None] = 0.0
    for i, j in accumulator:
        ti.atomic_max(max_density[None], accumulator[i, j][3])

@ti.kernel
def apply_tone_mapping_kernel(
    accumulator: ti.template(), pixels: ti.template(), max_density: ti.template(),
    min_density: ti.f32, noise_falloff: ti.f32, vibrance: ti.f32, brightness: ti.f32, gamma: ti.f32
):
    max_d = max_density[None]
    safe_max = ti.math.max(max_d - min_density + 1.0, 1.0)
    log_max = ti.math.max(ti.math.log(safe_max), 1e-5)
    
    for i, j in accumulator:
        dens = accumulator[i, j][3]
        if max_d > min_density and dens >= min_density:
            effective_dens = dens - min_density + 1.0
            alpha = ti.math.max(ti.math.log(effective_dens) / log_max, 0.0)
            alpha = ti.math.pow(alpha, noise_falloff)
            
            r = accumulator[i, j][0] / dens
            g = accumulator[i, j][1] / dens
            b = accumulator[i, j][2] / dens
            
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            r = luminance + (r - luminance) * vibrance
            g = luminance + (g - luminance) * vibrance
            b = luminance + (b - luminance) * vibrance
            
            final_r = ti.math.pow(ti.math.max(r * alpha * brightness, 0.0), 1.0 / gamma)
            final_g = ti.math.pow(ti.math.max(g * alpha * brightness, 0.0), 1.0 / gamma)
            final_b = ti.math.pow(ti.math.max(b * alpha * brightness, 0.0), 1.0 / gamma)
            
            pixels[i, j] = ti.math.clamp(ti.math.vec3(final_r, final_g, final_b), 0.0, 1.0)
        else:
            pixels[i, j] = ti.math.vec3(0.0, 0.0, 0.0)


# =====================================================================
# DATA / PARSING MODULE
# =====================================================================
class FlameParser:
    """Handles Reading and Writing Apophysis .flame files"""
    
    @staticmethod
    def create_sample_flame_data():
        """Returns default dummy data for initialization."""
        xforms = [
            {'coefs': [0.8, 0.3, -0.3, 0.8, 0.0, 0.0], 'weight': 0.33, 'color_idx': 0.0, 'v_linear': 0.2, 'v_swirl': 0.8},
            {'coefs': [0.5, 0.0, 0.0, 0.5, 1.0, 0.5], 'weight': 0.33, 'color_idx': 0.5, 'v_spherical': 1.0},
            {'coefs': [0.3, 0.5, -0.5, 0.3, -1.0, -0.5], 'weight': 0.34, 'color_idx': 1.0, 'v_horseshoe': 0.8}
        ]
        # Guarantee all variables exist
        for xf in xforms:
            for v in SUPPORTED_VARS:
                if f'v_{v}' not in xf: xf[f'v_{v}'] = 0.0
        
        palette_data = [(i/255.0, (i/255.0)**2, 0.0) for i in range(256)]
        camera = {"scale": 150.0, "x": 0.0, "y": 0.0}
        return camera, xforms, palette_data

    @staticmethod
    def load(filepath):
        print(f"Loading {filepath}...")
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            flame = root.find('flame')
            if flame is None: flame = root if root.tag == 'flame' else None
            if flame is None: raise ValueError("Invalid .flame file format.")

            camera = {"scale": 100.0, "x": 0.0, "y": 0.0}
            if 'scale' in flame.attrib: camera['scale'] = float(flame.attrib['scale'])
            if 'center' in flame.attrib:
                cx, cy = map(float, flame.attrib['center'].split())
                camera['x'] = cx; camera['y'] = cy

            palette_data = []
            palette_tag = flame.find('palette')
            if palette_tag is not None and palette_tag.text:
                text = palette_tag.text.replace("\n", "").replace(" ", "").strip()
                for i in range(256):
                    if i*6+6 <= len(text):
                        hex_str = text[i*6 : i*6+6]
                        palette_data.append((int(hex_str[0:2], 16)/255.0, int(hex_str[2:4], 16)/255.0, int(hex_str[4:6], 16)/255.0))
            if len(palette_data) < 256:
                palette_data = [(i/255.0, (i/255.0)**2, 0.0) for i in range(256)]

            xforms = []
            total_weight = 0.0
            for xform_tag in flame.findall('xform'):
                coefs = list(map(float, xform_tag.attrib.get('coefs', '1 0 0 1 0 0').split()))
                weight = float(xform_tag.attrib.get('weight', 1.0))
                color_idx = float(xform_tag.attrib.get('color', 0.0))
                total_weight += weight
                
                xf_data = {'coefs': coefs, 'weight': weight, 'color_idx': color_idx}
                for v in SUPPORTED_VARS: xf_data[f'v_{v}'] = float(xform_tag.attrib.get(v, 0.0))
                xforms.append(xf_data)
                
            for xf in xforms: xf['weight'] /= max(total_weight, 1e-10)
            
            return camera, xforms, palette_data
            
        except Exception as e:
            print(f"Error loading flame file: {e}")
            return None, None, None

    @staticmethod
    def save(filepath, camera, xforms, palette_data, width, height):
        print(f"Saving flame parameters to {filepath}...")
        root = ET.Element("flames", name="Custom_Flames")
        flame = ET.SubElement(root, "flame", name="Exported_Render")
        flame.set("version", "Apophysis 7x")
        flame.set("size", f'{width} {height}')
        flame.set("center", f'{camera["x"]} {camera["y"]}')
        flame.set("scale", str(camera["scale"]))

        for xf in xforms:
            attribs = {
                "weight": str(round(xf['weight'], 6)), "color": str(round(xf['color_idx'], 6)),
                "coefs": " ".join(str(round(c, 6)) for c in xf['coefs'])
            }
            for v in SUPPORTED_VARS:
                val = xf.get(f'v_{v}', 0.0)
                if val != 0.0: attribs[v] = str(round(val, 6))
            ET.SubElement(flame, "xform", attribs)

        pal = ET.SubElement(flame, "palette", count="256")
        pal_str = ""
        for idx, color in enumerate(palette_data):
            if idx > 0 and idx % 8 == 0: pal_str += "\n"
            pal_str += f"{int(max(0, min(1, color[0]))*255):02X}{int(max(0, min(1, color[1]))*255):02X}{int(max(0, min(1, color[2]))*255):02X}"
        pal_str += "\n"
        pal.text = pal_str

        tree = ET.ElementTree(root)
        if hasattr(ET, "indent"): ET.indent(tree, space="  ")
        tree.write(filepath, encoding="utf-8", xml_declaration=True)


# =====================================================================
# RENDERER ENGINE
# =====================================================================
class FlameRenderer:
    """Core Render Engine managing GPU VRAM, parameters, and executing Taichi Kernels."""
    def __init__(self, config=None):
        self.config = config if config else DEFAULT_CONFIG.copy()
        self.camera, self.xforms, self.palette_data = FlameParser.create_sample_flame_data()
        
        # Dynamic Taichi Fields
        self.accumulator = None
        self.pixels = None
        self.max_density = ti.field(dtype=ti.f32, shape=())
        self.init_fields()

    def update_config(self, key, value):
        self.config[key] = value
        if key in ["FINAL_WIDTH", "FINAL_HEIGHT", "OVERSAMPLE"]:
            self.init_fields()

    def init_fields(self):
        w = self.config["FINAL_WIDTH"] * self.config["OVERSAMPLE"]
        h = self.config["FINAL_HEIGHT"] * self.config["OVERSAMPLE"]
        self.accumulator = ti.Vector.field(4, dtype=ti.f32, shape=(w, h))
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))

    def load_flame(self, filepath):
        c, x, p = FlameParser.load(filepath)
        if c:
            self.camera, self.xforms, self.palette_data = c, x, p
            return True
        return False

    def render(self):
        w = self.config["FINAL_WIDTH"] * self.config["OVERSAMPLE"]
        h = self.config["FINAL_HEIGHT"] * self.config["OVERSAMPLE"]

        # Push structures to VRAM
        num_xfs = len(self.xforms)
        d_xforms = XFormStruct.field(shape=(num_xfs,))
        d_palette = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        
        for i, xf in enumerate(self.xforms):
            d_xforms[i].a, d_xforms[i].b, d_xforms[i].c, d_xforms[i].d, d_xforms[i].e, d_xforms[i].f = xf['coefs']
            d_xforms[i].weight = xf['weight']
            d_xforms[i].color_idx = xf['color_idx']
            for v in SUPPORTED_VARS:
                setattr(d_xforms[i], f'v_{v}', xf[f'v_{v}'])
                
        for i in range(256):
            d_palette[i] = ti.Vector(self.palette_data[i])
            
        self.accumulator.fill(0)
        
        # Execute Kernels (Passing Fields as Templates)
        render_flame_kernel(
            num_xfs, d_xforms, d_palette, self.accumulator,
            self.camera['scale'], self.camera['x'], self.camera['y'],
            w, h, self.config["ITERATIONS"], self.config["NUM_THREADS"], self.config["BURN_IN"],
            self.config["FILTER_RADIUS"], self.config["FILTER_WEIGHT_CENTER"], self.config["FILTER_WEIGHT_EDGE"]
        )
        
        find_max_density_kernel(self.accumulator, self.max_density)
        
        apply_tone_mapping_kernel(
            self.accumulator, self.pixels, self.max_density,
            self.config["MIN_DENSITY"], self.config["NOISE_FALLOFF"], 
            self.config["VIBRANCE"], self.config["BRIGHTNESS"], self.config["GAMMA"]
        )
        ti.sync()

    def get_dpg_texture(self):
        """Returns flat RGBA numpy array required by DearPyGui."""
        img_np = self.pixels.to_numpy()
        img_np = np.swapaxes(img_np, 0, 1)[::-1, :, :]        
        h, w = img_np.shape[0], img_np.shape[1]
        rgba_np = np.ones((h, w, 4), dtype=np.float32)
        rgba_np[:, :, :3] = img_np
        return rgba_np.flatten()

    def get_pil_image(self):
        """Extracts result from Taichi, processes it, and returns a PIL Image."""
        img_np = self.pixels.to_numpy()
        img_np = np.swapaxes(img_np, 0, 1)[::-1, :, :]        
        img_uint8 = (img_np * 255.0).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'RGB')

        # Apply cinematic post-processing if high-quality / oversampled
        if self.config["OVERSAMPLE"] > 1:
            print("Applying Anti-Aliasing Downscale...")
            img = img.resize((self.config["FINAL_WIDTH"], self.config["FINAL_HEIGHT"]), Image.Resampling.LANCZOS)
            
            print("Applying Noise Reduction Filter...")
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            print("Applying Bloom and Color Grade...")
            blur_radius = self.config["FINAL_WIDTH"] * 0.015
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            screened_img = ImageChops.screen(img, blurred_img)
            img = Image.blend(img, screened_img, self.config["BLOOM_INTENSITY"])
            img = ImageEnhance.Contrast(img).enhance(1.15) 
            img = ImageEnhance.Color(img).enhance(1.1)   
        
        return img


# =====================================================================
# USER INTERFACE MODULE
# =====================================================================
class FlameGUI:
    """Manages User Interface, state synchronization, and render loop."""
    def __init__(self, renderer: FlameRenderer):
        self.renderer = renderer
        self.texture_tag = "viewport_texture"
        self.needs_update = True
        
    def trigger_update(self, sender=None, app_data=None, user_data=None):
        self.needs_update = True

    def update_xform_coef(self, sender, app_data, user_data):
        x_idx, c_idx = user_data
        self.renderer.xforms[x_idx]['coefs'][c_idx] = app_data
        self.needs_update = True

    def update_xform_var(self, sender, app_data, user_data):
        x_idx, var_name = user_data
        self.renderer.xforms[x_idx][var_name] = app_data
        self.needs_update = True
        
    def update_xform_base(self, sender, app_data, user_data):
        x_idx, key = user_data
        self.renderer.xforms[x_idx][key] = app_data
        self.needs_update = True

    def toggle_hq(self, sender, app_data, user_data):
        if app_data:
            self.renderer.update_config("ITERATIONS", 50_000)
            self.renderer.update_config("OVERSAMPLE", 2)
            self.renderer.update_config("MIN_DENSITY", 12.0)
            dpg.set_value("status_text", "Status: High Quality (Slow)")
        else:
            self.renderer.update_config("ITERATIONS", 500)
            self.renderer.update_config("OVERSAMPLE", 1)
            self.renderer.update_config("MIN_DENSITY", 1.0)
            dpg.set_value("status_text", "Status: Real-time Preview")
        self.needs_update = True

    def build_transforms_ui(self):
        dpg.delete_item("transforms_container", children_only=True)
        for i, xf in enumerate(self.renderer.xforms):
            with dpg.collapsing_header(label=f"Transform {i+1}", parent="transforms_container"):
                dpg.add_slider_float(label="Weight", default_value=xf['weight'], min_value=0.0, max_value=2.0, callback=self.update_xform_base, user_data=(i, 'weight'))
                dpg.add_slider_float(label="Color Idx", default_value=xf['color_idx'], min_value=0.0, max_value=1.0, callback=self.update_xform_base, user_data=(i, 'color_idx'))
                
                dpg.add_text("Affine Coefficients")
                coef_labels = ['a (Scale X)', 'b (Shear X)', 'c (Shear Y)', 'd (Scale Y)', 'e (Translate X)', 'f (Translate Y)']
                for c_idx, label in enumerate(coef_labels):
                    dpg.add_slider_float(label=label, default_value=xf['coefs'][c_idx], min_value=-2.0, max_value=2.0, callback=self.update_xform_coef, user_data=(i, c_idx))
                
                dpg.add_text("Variations")
                for v in SUPPORTED_VARS:
                    if xf.get(f'v_{v}', 0.0) > 0.0 or v in ['linear', 'spherical', 'swirl', 'horseshoe']:
                        dpg.add_slider_float(label=v, default_value=xf.get(f'v_{v}', 0.0), min_value=-2.0, max_value=2.0, callback=self.update_xform_var, user_data=(i, f'v_{v}'))

    def load_callback(self, sender, app_data):
        if self.renderer.load_flame(app_data['file_path_name']):
            dpg.set_value("cam_scale_slider", self.renderer.camera["scale"])
            dpg.set_value("cam_x_slider", self.renderer.camera["x"])
            dpg.set_value("cam_y_slider", self.renderer.camera["y"])
            self.build_transforms_ui()
            self.needs_update = True

    def save_image_callback(self):
        img = self.renderer.get_pil_image()
        img.save("editor_render.png")
        print("Image saved to editor_render.png")

    def save_xml_callback(self):
        FlameParser.save("saved_fractal.flame", self.renderer.camera, self.renderer.xforms, self.renderer.palette_data, self.renderer.config["FINAL_WIDTH"], self.renderer.config["FINAL_HEIGHT"])

    def run(self):
        dpg.create_context()
        
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.load_callback, tag="file_dialog_id", width=700, height=400):
            dpg.add_file_extension(".flame", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

        with dpg.texture_registry(show=False):
            initial_data = np.zeros(self.renderer.config["FINAL_WIDTH"] * self.renderer.config["FINAL_HEIGHT"] * 4, dtype=np.float32)
            dpg.add_dynamic_texture(width=self.renderer.config["FINAL_WIDTH"], height=self.renderer.config["FINAL_HEIGHT"], default_value=initial_data, tag=self.texture_tag)

        with dpg.window(label="Apophysis Editor", width=400, height=800, pos=(0,0)):
            dpg.add_text("Rendering Settings")
            dpg.add_checkbox(label="High Quality Render Mode", callback=self.toggle_hq)
            dpg.add_text("Status: Real-time Preview", tag="status_text", color=(100, 255, 100))
            
            dpg.add_slider_float(label="Brightness", default_value=self.renderer.config["BRIGHTNESS"], min_value=0.1, max_value=5.0, callback=lambda s,a,u: (self.renderer.update_config("BRIGHTNESS", a), self.trigger_update()))
            dpg.add_slider_float(label="Vibrance", default_value=self.renderer.config["VIBRANCE"], min_value=0.0, max_value=3.0, callback=lambda s,a,u: (self.renderer.update_config("VIBRANCE", a), self.trigger_update()))
            
            dpg.add_separator()
            dpg.add_text("Camera")
            dpg.add_slider_float(label="Scale (Zoom)", default_value=self.renderer.camera["scale"], min_value=10.0, max_value=1000.0, callback=lambda s,a,u: (self.renderer.camera.update({"scale": a}), self.trigger_update()), tag="cam_scale_slider")
            dpg.add_slider_float(label="X Offset", default_value=self.renderer.camera["x"], min_value=-5.0, max_value=5.0, callback=lambda s,a,u: (self.renderer.camera.update({"x": a}), self.trigger_update()), tag="cam_x_slider")
            dpg.add_slider_float(label="Y Offset", default_value=self.renderer.camera["y"], min_value=-5.0, max_value=5.0, callback=lambda s,a,u: (self.renderer.camera.update({"y": a}), self.trigger_update()), tag="cam_y_slider")

            dpg.add_separator()
            dpg.add_text("Transforms")
            with dpg.group(tag="transforms_container"): pass
            self.build_transforms_ui()

            dpg.add_separator()
            dpg.add_button(label="Open Flame File (.flame)", callback=lambda: dpg.show_item("file_dialog_id"))
            dpg.add_button(label="Save Final Render (.png)", callback=self.save_image_callback)
            dpg.add_button(label="Save Flame File (.flame)", callback=self.save_xml_callback)

        with dpg.window(label="Viewport", width=self.renderer.config["FINAL_WIDTH"]+20, height=self.renderer.config["FINAL_HEIGHT"]+40, pos=(410, 0)):
            dpg.add_image(self.texture_tag)

        dpg.create_viewport(title='Refactored Apophysis Editor', width=1250, height=850)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Render Loop
        while dpg.is_dearpygui_running():
            if self.needs_update:
                self.renderer.render()
                dpg.set_value(self.texture_tag, self.renderer.get_dpg_texture())
                self.needs_update = False
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

# =====================================================================
# COMMAND LINE INTERFACE (ROUTER)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Apophysis Editor and High-Quality CLI Renderer.")
    parser.add_argument("--cli", action="store_true", help="Run in headless command-line mode (requires -i).")
    parser.add_argument("-i", "--input", type=str, help="Path to input .flame file.")
    parser.add_argument("-o", "--output", type=str, default="cli_render.png", help="Path for output image.")
    parser.add_argument("-z", "--zoom", type=float, default=1.0, help="Zoom multiplier for CLI mode.")
    args = parser.parse_args()

    if args.cli or args.input:
        print("=== CLI Render Mode Initialized ===")
        if not args.input or not os.path.exists(args.input):
            print(f"Error: Could not find flame file at '{args.input}'. Please provide a valid file via -i.")
            exit(1)
            
        # Configure Renderer for High Quality
        config = DEFAULT_CONFIG.copy()
        config["ITERATIONS"] = 50_000
        config["OVERSAMPLE"] = 2
        config["MIN_DENSITY"] = 12.0
        
        # Override output resolution if you desire higher res for CLI (e.g., 4K)
        config["FINAL_WIDTH"] = 1920
        config["FINAL_HEIGHT"] = 1080
        
        renderer = FlameRenderer(config)
        renderer.load_flame(args.input)
        renderer.camera["scale"] *= args.zoom  # Apply zoom factor
        
        print(f"Starting compute shader: {config['NUM_THREADS']} threads x {config['ITERATIONS']} iterations...")
        start_time = time.time()
        renderer.render()
        print(f"Compute finished in {time.time() - start_time:.2f} seconds.")
        
        img = renderer.get_pil_image() # This now safely includes cinematic post-processing!
        img.save(args.output)
        print(f"Success! Image saved to {args.output}")
        
    else:
        print("=== GUI Editor Mode Initialized ===")
        renderer = FlameRenderer()
        gui = FlameGUI(renderer)
        gui.run()
