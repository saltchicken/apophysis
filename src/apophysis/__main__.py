import os
import math
import argparse
import taichi as ti

# Import the refactored renderer logic
from .renderer import ApophysisRenderer


def generate_sample_flame(filepath):
    xml_content = """<flames>
<flame name="Cosmic_Flower" version="Apophysis 7x" size="1920 1080" center="0 0" scale="250">
    <xform weight="0.5" color="0.0" swirl="0.8" linear="0.2" coefs="0.8 0.3 -0.3 0.8 0 0" />
    <xform weight="0.5" color="0.5" spherical="1.0" coefs="0.5 0 0 0.5 1.0 0.5" />
    <xform weight="0.5" color="1.0" horseshoe="0.8" coefs="0.3 0.5 -0.5 0.3 -1.0 -0.5" />
    <palette count="256">\n"""
    for i in range(256):
        r = int((math.sin(i * 0.05) * 0.5 + 0.5) * 255)
        g = int((math.sin(i * 0.05 + 2) * 0.5 + 0.5) * 255)
        b = int((math.sin(i * 0.05 + 4) * 0.5 + 0.5) * 255)
        xml_content += f"{r:02x}{g:02x}{b:02x}\n"
    xml_content += """    </palette>\n</flame>\n</flames>"""

    with open(filepath, "w") as f:
        f.write(xml_content)
    print(f"Generated sample flame file at {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Apophysis Flame Renderer"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="sample_fractal.flame",
        help="Path to .flame file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="render_output.png",
        help="Output image path",
    )

    # Exposing parameters dynamically to the user
    parser.add_argument(
        "-W", "--width", type=int, default=1920, help="Final output width"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=1080, help="Final output height"
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        help="SSAA multiplier (e.g. 2 means 4K down to 1080p)",
    )
    parser.add_argument(
        "-z", "--zoom", type=float, default=1.0, help="Camera zoom multiplier"
    )

    # Quality & Performance flags
    parser.add_argument(
        "--threads", type=int, default=200_000, help="Number of parallel GPU threads"
    )
    parser.add_argument(
        "--iterations", type=int, default=50_000, help="Total iterations per thread"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="Iterations per GPU execution to prevent crashing",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input) and args.input == "sample_fractal.flame":
        generate_sample_flame(args.input)
    elif not os.path.exists(args.input):
        print(f"Error: The file '{args.input}' does not exist.")
        exit(1)

    # Initialize Taichi
    ti.init(arch=ti.cuda)

    try:
        # Initialize Renderer with Dynamic Attributes
        renderer = ApophysisRenderer(
            final_width=args.width,
            final_height=args.height,
            oversample=args.oversample,
            num_threads=args.threads,
            iterations=args.iterations,
            batch_size=args.batch_size,
        )

        # Load File and Render
        renderer.load_flame(args.input, zoom_multiplier=args.zoom)
        renderer.render_to_image(args.output)

    except Exception as e:
        print(f"Error rendering fractal: {e}")


if __name__ == "__main__":
    main()
