import math
import time
import xml.etree.ElementTree as ET
import taichi as ti
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageChops

# Data structure mapping to hold our Affine Transforms and Variation weights in VRAM
XFormStruct = ti.types.struct(
    a=ti.f32,
    b=ti.f32,
    c=ti.f32,
    d=ti.f32,
    e=ti.f32,
    f=ti.f32,
    weight=ti.f32,
    color_idx=ti.f32,
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
    v_gaussian_blur=ti.f32,
)


@ti.data_oriented
class ApophysisRenderer:
    def __init__(
        self,
        final_width=1920,
        final_height=1080,
        oversample=2,
        num_threads=200_000,
        iterations=50_000,
        batch_size=5_000,
        gamma=2.2,
        brightness=2.0,
        vibrance=1.4,
        bloom_intensity=0.4,
        min_density=12.0,
        noise_falloff=2.5,
    ):
        self.final_width = final_width
        self.final_height = final_height
        self.oversample = oversample
        self.width = final_width * oversample
        self.height = final_height * oversample

        self.num_threads = num_threads
        self.iterations = iterations
        self.batch_size = batch_size
        self.burn_in = 50

        # Rendering Params
        self.gamma = gamma
        self.brightness = brightness
        self.vibrance = vibrance
        self.bloom_intensity = bloom_intensity
        self.min_density = min_density
        self.noise_falloff = noise_falloff

        self.filter_radius = 1
        self.filter_weight_center = 1.0
        self.filter_weight_edge = 0.5

        # Camera & Data
        self.camera = {"scale": 100.0, "x": 0.0, "y": 0.0}
        self.xforms_data = []
        self.palette_data = []
        self.num_xfs = 0

        self.supported_vars = [
            "linear",
            "sinusoidal",
            "spherical",
            "swirl",
            "horseshoe",
            "polar",
            "handkerchief",
            "heart",
            "disc",
            "spiral",
            "hyperbolic",
            "diamond",
            "ex",
            "julia",
            "bent",
            "waves",
            "fisheye",
            "popcorn",
            "exponential",
            "power",
            "cosine",
            "rings",
            "fan",
            "eyefish",
            "bubble",
            "cylinder",
            "noise",
            "blur",
            "gaussian_blur",
        ]

        # Dynamically allocate Taichi fields based on chosen resolution
        self.accumulator = ti.Vector.field(
            4, dtype=ti.f32, shape=(self.width, self.height)
        )
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.max_density = ti.field(dtype=ti.f32, shape=())

        # We will allocate these after parsing the flame
        self.d_xforms = None
        self.d_palette = ti.Vector.field(3, dtype=ti.f32, shape=(256,))

    def load_flame(self, flame_path, zoom_multiplier=1.0):
        print(f"Parsing {flame_path}...")
        tree = ET.parse(flame_path)
        root = tree.getroot()
        flame = (
            root.find("flame")
            if root.find("flame") is not None
            else (root if root.tag == "flame" else None)
        )

        if flame is None:
            raise ValueError("Invalid .flame file format.")

        if "scale" in flame.attrib:
            self.camera["scale"] = (
                float(flame.attrib["scale"]) * self.oversample * zoom_multiplier
            )
        if "center" in flame.attrib:
            cx, cy = map(float, flame.attrib["center"].split())
            self.camera["x"] = cx
            self.camera["y"] = cy

        # Parse Palette
        palette_tag = flame.find("palette")
        if palette_tag is not None and palette_tag.text:
            text = palette_tag.text.replace("\n", "").replace(" ", "").strip()
            for i in range(256):
                if i * 6 + 6 <= len(text):
                    hex_str = text[i * 6 : i * 6 + 6]
                    self.palette_data.append(
                        (
                            int(hex_str[0:2], 16) / 255.0,
                            int(hex_str[2:4], 16) / 255.0,
                            int(hex_str[4:6], 16) / 255.0,
                        )
                    )

        if len(self.palette_data) < 256:
            self.palette_data = [(i / 255.0, (i / 255.0) ** 2, 0.0) for i in range(256)]

        for i in range(256):
            self.d_palette[i] = ti.Vector(self.palette_data[i])

        # Parse Transforms
        total_weight = 0.0
        for xform_tag in flame.findall("xform"):
            coefs = list(
                map(float, xform_tag.attrib.get("coefs", "1 0 0 1 0 0").split())
            )
            weight = float(xform_tag.attrib.get("weight", 1.0))
            color_idx = float(xform_tag.attrib.get("color", 0.0))
            total_weight += weight

            xf_data = {"coefs": coefs, "weight": weight, "color_idx": color_idx}
            for v in self.supported_vars:
                xf_data[f"v_{v}"] = float(xform_tag.attrib.get(v, 0.0))
            self.xforms_data.append(xf_data)

        self.num_xfs = len(self.xforms_data)
        self.d_xforms = XFormStruct.field(shape=(self.num_xfs,))

        for i, xf in enumerate(self.xforms_data):
            xf["weight"] /= total_weight
            (
                self.d_xforms[i].a,
                self.d_xforms[i].b,
                self.d_xforms[i].c,
                self.d_xforms[i].d,
                self.d_xforms[i].e,
                self.d_xforms[i].f,
            ) = xf["coefs"]
            self.d_xforms[i].weight = xf["weight"]
            self.d_xforms[i].color_idx = xf["color_idx"]
            for v in self.supported_vars:
                setattr(self.d_xforms[i], f"v_{v}", xf[f"v_{v}"])

        print(f"Parsed {self.num_xfs} transforms successfully.")

    @ti.kernel
    def _render_batch_kernel(self, batch_iters: ti.i32):
        for _ in range(self.num_threads):
            x = ti.random(ti.f32) * 2.0 - 1.0
            y = ti.random(ti.f32) * 2.0 - 1.0
            c = ti.random(ti.f32)

            for step in range(batch_iters + self.burn_in):
                rand_choice = ti.random(ti.f32)
                cumulative_weight = 0.0
                chosen_idx = 0

                for j in range(self.num_xfs):
                    cumulative_weight += self.d_xforms[j].weight
                    if rand_choice <= cumulative_weight:
                        chosen_idx = j
                        break

                xf = self.d_xforms[chosen_idx]
                nx = xf.a * x + xf.c * y + xf.e
                ny = xf.b * x + xf.d * y + xf.f

                final_x = 0.0
                final_y = 0.0
                r2 = nx * nx + ny * ny
                r = ti.math.sqrt(r2)
                theta = ti.math.atan2(ny, nx)

                # Variations
                if xf.v_linear > 0.0:
                    final_x += xf.v_linear * nx
                    final_y += xf.v_linear * ny
                if xf.v_sinusoidal > 0.0:
                    final_x += xf.v_sinusoidal * ti.math.sin(nx)
                    final_y += xf.v_sinusoidal * ti.math.sin(ny)
                if xf.v_spherical > 0.0:
                    r2_safe = ti.math.max(r2, 1e-10)
                    final_x += xf.v_spherical * (nx / r2_safe)
                    final_y += xf.v_spherical * (ny / r2_safe)
                if xf.v_swirl > 0.0:
                    sin_r2 = ti.math.sin(r2)
                    cos_r2 = ti.math.cos(r2)
                    final_x += xf.v_swirl * (nx * sin_r2 - ny * cos_r2)
                    final_y += xf.v_swirl * (nx * cos_r2 + ny * sin_r2)
                if xf.v_horseshoe > 0.0:
                    r_safe = ti.math.max(r, 1e-10)
                    final_x += xf.v_horseshoe * ((nx - ny) * (nx + ny)) / r_safe
                    final_y += xf.v_horseshoe * (2.0 * nx * ny) / r_safe
                if xf.v_polar > 0.0:
                    final_x += xf.v_polar * (theta / ti.math.pi)
                    final_y += xf.v_polar * (r - 1.0)
                if xf.v_handkerchief > 0.0:
                    final_x += xf.v_handkerchief * r * ti.math.sin(theta + r)
                    final_y += xf.v_handkerchief * r * ti.math.cos(theta - r)
                if xf.v_heart > 0.0:
                    final_x += xf.v_heart * r * ti.math.sin(theta * r)
                    final_y += xf.v_heart * -r * ti.math.cos(theta * r)
                if xf.v_disc > 0.0:
                    final_x += (
                        xf.v_disc * (theta / ti.math.pi) * ti.math.sin(ti.math.pi * r)
                    )
                    final_y += (
                        xf.v_disc * (theta / ti.math.pi) * ti.math.cos(ti.math.pi * r)
                    )
                if xf.v_spiral > 0.0:
                    r_safe = ti.math.max(r, 1e-10)
                    final_x += (
                        xf.v_spiral * (ti.math.cos(theta) + ti.math.sin(r)) / r_safe
                    )
                    final_y += (
                        xf.v_spiral * (ti.math.sin(theta) - ti.math.cos(r)) / r_safe
                    )
                if xf.v_hyperbolic > 0.0:
                    r_safe = ti.math.max(r, 1e-10)
                    final_x += xf.v_hyperbolic * ti.math.sin(theta) / r_safe
                    final_y += xf.v_hyperbolic * r * ti.math.cos(theta)
                if xf.v_diamond > 0.0:
                    final_x += xf.v_diamond * ti.math.sin(theta) * ti.math.cos(r)
                    final_y += xf.v_diamond * ti.math.cos(theta) * ti.math.sin(r)
                if xf.v_ex > 0.0:
                    n0 = ti.math.sin(theta + r)
                    n1 = ti.math.cos(theta - r)
                    m0 = n0 * n0 * n0
                    m1 = n1 * n1 * n1
                    final_x += xf.v_ex * r * (m0 + m1)
                    final_y += xf.v_ex * r * (m0 - m1)
                if xf.v_julia > 0.0:
                    r_julia = ti.math.sqrt(r)
                    theta_julia = theta * 0.5 + ti.math.pi * (ti.random(ti.i32) % 2)
                    final_x += xf.v_julia * r_julia * ti.math.cos(theta_julia)
                    final_y += xf.v_julia * r_julia * ti.math.sin(theta_julia)
                if xf.v_bent > 0.0:
                    bent_x = nx
                    bent_y = ny
                    if nx < 0.0 and ny >= 0.0:
                        bent_x = 2.0 * nx
                    elif nx >= 0.0 and ny < 0.0:
                        bent_y = ny * 0.5
                    elif nx < 0.0 and ny < 0.0:
                        bent_x = 2.0 * nx
                        bent_y = ny * 0.5
                    final_x += xf.v_bent * bent_x
                    final_y += xf.v_bent * bent_y
                if xf.v_waves > 0.0:
                    final_x += xf.v_waves * (
                        nx + xf.b * ti.math.sin(ny / (xf.c * xf.c + 1e-10))
                    )
                    final_y += xf.v_waves * (
                        ny + xf.e * ti.math.sin(nx / (xf.f * xf.f + 1e-10))
                    )
                if xf.v_fisheye > 0.0:
                    r_fisheye = 2.0 / (r + 1.0)
                    final_x += xf.v_fisheye * r_fisheye * ny
                    final_y += xf.v_fisheye * r_fisheye * nx
                if xf.v_popcorn > 0.0:
                    final_x += xf.v_popcorn * (
                        nx + xf.c * ti.math.sin(ti.math.tan(3.0 * ny))
                    )
                    final_y += xf.v_popcorn * (
                        ny + xf.f * ti.math.sin(ti.math.tan(3.0 * nx))
                    )
                if xf.v_exponential > 0.0:
                    exp_nx = ti.math.exp(nx - 1.0)
                    final_x += xf.v_exponential * exp_nx * ti.math.cos(ti.math.pi * ny)
                    final_y += xf.v_exponential * exp_nx * ti.math.sin(ti.math.pi * ny)
                if xf.v_power > 0.0:
                    r_safe = ti.math.max(r, 1e-10)
                    pow_theta = ti.math.pow(r_safe, ti.math.sin(theta))
                    final_x += xf.v_power * pow_theta * ti.math.cos(theta)
                    final_y += xf.v_power * pow_theta * ti.math.sin(theta)
                if xf.v_cosine > 0.0:
                    exp_ny = ti.math.exp(ny)
                    exp_neg_ny = ti.math.exp(-ny)
                    cosh_ny = 0.5 * (exp_ny + exp_neg_ny)
                    sinh_ny = 0.5 * (exp_ny - exp_neg_ny)
                    final_x += xf.v_cosine * ti.math.cos(ti.math.pi * nx) * cosh_ny
                    final_y += xf.v_cosine * -ti.math.sin(ti.math.pi * nx) * sinh_ny
                if xf.v_rings > 0.0:
                    dx = xf.c * xf.c + 1e-10
                    r_rings = ((r + dx) % (2.0 * dx)) - dx + r * (1.0 - dx)
                    final_x += xf.v_rings * r_rings * ti.math.cos(theta)
                    final_y += xf.v_rings * r_rings * ti.math.sin(theta)
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
                if xf.v_eyefish > 0.0:
                    r_eyefish = 2.0 / (r + 1.0)
                    final_x += xf.v_eyefish * r_eyefish * nx
                    final_y += xf.v_eyefish * r_eyefish * ny
                if xf.v_bubble > 0.0:
                    r_bubble = 4.0 / (r2 + 4.0)
                    final_x += xf.v_bubble * r_bubble * nx
                    final_y += xf.v_bubble * r_bubble * ny
                if xf.v_cylinder > 0.0:
                    final_x += xf.v_cylinder * ti.math.sin(nx)
                    final_y += xf.v_cylinder * ny
                if xf.v_noise > 0.0:
                    rx = ti.random(ti.f32)
                    ry = ti.random(ti.f32) * 2.0 * ti.math.pi
                    final_x += xf.v_noise * rx * nx * ti.math.cos(ry)
                    final_y += xf.v_noise * rx * ny * ti.math.sin(ry)
                if xf.v_blur > 0.0:
                    blur_r = ti.random(ti.f32)
                    blur_theta = ti.random(ti.f32) * 2.0 * ti.math.pi
                    final_x += xf.v_blur * blur_r * ti.math.cos(blur_theta)
                    final_y += xf.v_blur * blur_r * ti.math.sin(blur_theta)
                if xf.v_gaussian_blur > 0.0:
                    u1 = ti.math.max(ti.random(ti.f32), 1e-10)
                    u2 = ti.random(ti.f32)
                    z0 = ti.math.sqrt(-2.0 * ti.math.log(u1)) * ti.math.cos(
                        2.0 * ti.math.pi * u2
                    )
                    z1 = ti.math.sqrt(-2.0 * ti.math.log(u1)) * ti.math.sin(
                        2.0 * ti.math.pi * u2
                    )
                    final_x += xf.v_gaussian_blur * z0
                    final_y += xf.v_gaussian_blur * z1

                x = final_x
                y = final_y
                c = (c + xf.color_idx) * 0.5

                if step > self.burn_in:
                    px_float = (x - self.camera["x"]) * self.camera["scale"] + (
                        self.width / 2.0
                    )
                    py_float = (y - self.camera["y"]) * self.camera["scale"] + (
                        self.height / 2.0
                    )
                    px = ti.cast(px_float, ti.i32)
                    py = ti.cast(py_float, ti.i32)

                    if (
                        -self.filter_radius <= px < self.width + self.filter_radius
                        and -self.filter_radius <= py < self.height + self.filter_radius
                    ):

                        pal_idx = ti.math.clamp(ti.cast(c * 255.0, ti.i32), 0, 255)
                        color = self.d_palette[pal_idx]

                        for dx in range(-self.filter_radius, self.filter_radius + 1):
                            for dy in range(
                                -self.filter_radius, self.filter_radius + 1
                            ):
                                splat_x = px + dx
                                splat_y = py + dy
                                if (
                                    0 <= splat_x < self.width
                                    and 0 <= splat_y < self.height
                                ):
                                    dist2 = dx * dx + dy * dy
                                    weight = 0.0
                                    if dist2 == 0:
                                        weight = self.filter_weight_center
                                    elif (
                                        dist2 <= self.filter_radius * self.filter_radius
                                    ):
                                        weight = self.filter_weight_edge / ti.math.sqrt(
                                            float(dist2)
                                        )

                                    if weight > 0.0:
                                        ti.atomic_add(
                                            self.accumulator[splat_x, splat_y][0],
                                            color[0] * weight,
                                        )
                                        ti.atomic_add(
                                            self.accumulator[splat_x, splat_y][1],
                                            color[1] * weight,
                                        )
                                        ti.atomic_add(
                                            self.accumulator[splat_x, splat_y][2],
                                            color[2] * weight,
                                        )
                                        ti.atomic_add(
                                            self.accumulator[splat_x, splat_y][3],
                                            weight,
                                        )

    @ti.kernel
    def _find_max_density_kernel(self):
        self.max_density[None] = 0.0
        for i, j in self.accumulator:
            ti.atomic_max(self.max_density[None], self.accumulator[i, j][3])

    @ti.kernel
    def _apply_tone_mapping_kernel(self):
        max_d = self.max_density[None]
        safe_max = ti.math.max(max_d - self.min_density + 1.0, 1.0)
        log_max = ti.math.max(ti.math.log(safe_max), 1e-5)

        for i, j in self.accumulator:
            dens = self.accumulator[i, j][3]
            if max_d > self.min_density and dens >= self.min_density:
                effective_dens = dens - self.min_density + 1.0
                alpha = ti.math.max(ti.math.log(effective_dens) / log_max, 0.0)
                alpha = ti.math.pow(alpha, self.noise_falloff)

                r = self.accumulator[i, j][0] / dens
                g = self.accumulator[i, j][1] / dens
                b = self.accumulator[i, j][2] / dens

                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                r = luminance + (r - luminance) * self.vibrance
                g = luminance + (g - luminance) * self.vibrance
                b = luminance + (b - luminance) * self.vibrance

                final_r = ti.math.pow(
                    ti.math.max(r * alpha * self.brightness, 0.0), 1.0 / self.gamma
                )
                final_g = ti.math.pow(
                    ti.math.max(g * alpha * self.brightness, 0.0), 1.0 / self.gamma
                )
                final_b = ti.math.pow(
                    ti.math.max(b * alpha * self.brightness, 0.0), 1.0 / self.gamma
                )

                self.pixels[i, j] = ti.math.clamp(
                    ti.math.vec3(final_r, final_g, final_b), 0.0, 1.0
                )
            else:
                self.pixels[i, j] = ti.math.vec3(0.0, 0.0, 0.0)

    def render_to_image(self, output_path):
        self.accumulator.fill(0)

        # Batching logic to prevent OS GPU driver timeout (TDR)
        num_batches = max(1, self.iterations // self.batch_size)
        total_iters = num_batches * self.batch_size

        print(
            f"Executing Compute Shader: {self.num_threads} threads * {total_iters} iterations."
        )
        print(f"Running in {num_batches} batches to maintain GPU stability...")

        start_time = time.time()
        for batch in range(num_batches):
            self._render_batch_kernel(self.batch_size)
            ti.sync()
            if num_batches > 1 and (batch + 1) % (max(1, num_batches // 10)) == 0:
                print(f"  Progress: {(batch + 1) / num_batches * 100:.0f}%")

        print(f"Compute finished in {time.time() - start_time:.2f} seconds.")

        print("Applying Log-Density Tone Mapping...")
        self._find_max_density_kernel()
        self._apply_tone_mapping_kernel()
        ti.sync()

        img_np = self.pixels.to_numpy()
        img_np = np.swapaxes(img_np, 0, 1)[::-1, :, :]
        img_uint8 = (img_np * 255.0).astype(np.uint8)
        img = Image.fromarray(img_uint8, "RGB")

        if self.oversample > 1:
            print(
                f"Downscaling to {self.final_width}x{self.final_height} for Anti-Aliasing..."
            )
            img = img.resize(
                (self.final_width, self.final_height), Image.Resampling.LANCZOS
            )

        print("Applying Median Filter & Cinematic Post-Processing...")
        img = img.filter(ImageFilter.MedianFilter(size=3))
        blur_radius = self.final_width * 0.015
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        screened_img = ImageChops.screen(img, blurred_img)
        img = Image.blend(img, screened_img, self.bloom_intensity)

        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Color(img).enhance(1.1)

        img.save(output_path)
        print(f"Success! Cleaned and polished fractal saved to {output_path}")
