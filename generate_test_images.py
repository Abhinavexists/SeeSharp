"""
Generate Test Images for ERSVR Model Testing
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import requests


def create_test_images_dir():
    os.makedirs('test_images', exist_ok=True)
    return 'test_images'


def generate_pattern_image(name, size=(128, 128)):
    height, width = size
    
    if name == 'checkerboard':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, 16):
            for j in range(0, width, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    img[i:i+16, j:j+16] = [255, 255, 255]
                else:
                    img[i:i+16, j:j+16] = [0, 0, 0]
    
    elif name == 'gradient':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                img[i, j] = [i * 255 // height, j * 255 // width, (i + j) * 255 // (height + width)]
    
    elif name == 'circles':
        img = np.zeros((height, width, 3), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if int(dist) % 20 < 10:
                    img[i, j] = [255, 100, 50]
                else:
                    img[i, j] = [50, 100, 255]
    
    elif name == 'text':
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "ERSVR", fill=(0, 0, 0), font=font)
        draw.text((10, 40), "Super", fill=(255, 0, 0), font=font)
        draw.text((10, 70), "Resolution", fill=(0, 0, 255), font=font)
        draw.text((10, 100), "Test", fill=(0, 255, 0), font=font)
        
        img = np.array(pil_img)
    
    elif name == 'noise':
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    else:  # default: stripes
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            if (i // 8) % 2 == 0:
                img[i, :] = [255, 255, 255]
            else:
                img[i, :] = [255, 0, 0]
    
    return img


def download_sample_image(url, filename):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False


def create_sample_sequence(base_img, output_dir, sequence_name):
    height, width = base_img.shape[:2]
    
    frames = []
    
    frames.append(base_img.copy())
    
    frame2 = base_img.copy().astype(np.float32)
    frame2 = np.clip(frame2 * 1.1, 0, 255).astype(np.uint8)
    frames.append(frame2)
    
    h, w = height, width
    M = cv2.getRotationMatrix2D((w/2, h/2), 2, 1.0)  # 2 degree rotation
    frame3 = cv2.warpAffine(base_img, M, (w, h))
    frames.append(frame3)
    
    seq_dir = os.path.join(output_dir, f"{sequence_name}_sequence")
    os.makedirs(seq_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(seq_dir, f"frame_{i+1}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    print(f"Created sequence: {seq_dir}")
    return seq_dir


def main():
    print("Generating test images for ERSVR model...")
    
    test_dir = create_test_images_dir()
    
    patterns = ['checkerboard', 'gradient', 'circles', 'text', 'noise', 'stripes']
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    print("\nGenerating pattern images...")
    for pattern in patterns:
        for size in sizes:
            img = generate_pattern_image(pattern, size)
            filename = os.path.join(test_dir, f"{pattern}_{size[0]}x{size[1]}.png")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Generated: {filename}")
    
    print("\nCreating sample sequences...")
    base_img = generate_pattern_image('circles', (128, 128))
    create_sample_sequence(base_img, test_dir, 'circles')
    
    base_img = generate_pattern_image('checkerboard', (128, 128))
    create_sample_sequence(base_img, test_dir, 'checkerboard')
    
    print("\nAttempting to download sample images...")
    sample_urls = [
        ("https://picsum.photos/200/200?random=1", "sample_natural_1.jpg"),
        ("https://picsum.photos/150/150?random=2", "sample_natural_2.jpg"),
        ("https://picsum.photos/300/200?random=3", "sample_natural_3.jpg"),
    ]
    
    for url, filename in sample_urls:
        filepath = os.path.join(test_dir, filename)
        download_sample_image(url, filepath)
    
    print(f"\nTest image generation complete!")
    print(f"Images saved to: {test_dir}")
    print(f"Found {len(os.listdir(test_dir))} test items")
    
    print("\nUsage examples:")
    print("Command line interface:")
    print(f"  python test_interface.py --image {test_dir}/circles_128x128.png")
    print(f"  python test_interface.py --frames {test_dir}/circles_sequence/frame_1.png {test_dir}/circles_sequence/frame_2.png {test_dir}/circles_sequence/frame_3.png")
    
    print("\nWeb interface:")
    print("  python web_interface.py")
    print("  Then open http://localhost:5000 in your browser")


if __name__ == '__main__':
    main() 