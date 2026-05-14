#!/usr/bin/env fish

echo "Generating animation frames..."
apophysis --frames 120 --rot-speed 0.02 --zoom-speed 1.005 -o anim_frame.png

echo "Compiling sequence into an MP4 video..."
ffmpeg -framerate 30 -i anim_frame_%04d.png -c:v libx264 -pix_fmt yuv420p apophysis_animation.mp4

echo "Cleaning up intermediate frames..."
rm anim_frame_*.png

echo "Animation successfully saved to apophysis_animation.mp4"