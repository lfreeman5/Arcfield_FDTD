import os
import imageio

# Directory containing the PNG files
directory = '.'  # Current directory

images = sorted([file for file in os.listdir() if file.startswith('iter_') and file.endswith('.png')],
               key=lambda x: int(x.split('_')[1].split('.')[0]))

mp4_filename = 'output.mp4'

# Create MP4 video
with imageio.get_writer(mp4_filename, format='FFMPEG', mode='I', fps=10) as writer:
    for image in images:
        frame = imageio.imread(image)
        writer.append_data(frame)

print(f'MP4 video saved as {mp4_filename}')

# Create GIF filename
gif_filename = 'output.gif'

# Create GIF
with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
    for image in images:
        frame = imageio.imread(image)
        writer.append_data(frame)

print(f'GIF saved as {gif_filename}')