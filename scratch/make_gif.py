import os
import imageio

def convert_mp4_to_gif(mp4_path, gif_path, max_frames=90, skip_frames=1):
    if not os.path.exists(mp4_path):
        print(f"Error: source video not found at {mp4_path}")
        return
        
    print(f"Converting {mp4_path} to GIF...")
    reader = imageio.get_reader(mp4_path)
    fps = reader.get_meta_data().get('fps', 30.0)
    
    frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        if i % skip_frames == 0:
            frames.append(frame)
            
    # Save frames as a looping GIF
    duration = (1.0 / fps) * skip_frames
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)
    print(f"[OK] Looping GIF saved to {gif_path}")

if __name__ == "__main__":
    src_mp4 = "videos/corridor_trained.mp4"
    dest_gif = "videos/corridor_showcase.gif"
    
    convert_mp4_to_gif(src_mp4, dest_gif, max_frames=120, skip_frames=2)
