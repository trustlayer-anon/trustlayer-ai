import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import random
import hashlib

def generate_fallback_pattern(h, w, pattern_id, rotation):
    pattern = np.zeros((h, w))
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    angle = np.arctan2(Y - cy, X - cx)
    
    rot_rad = np.deg2rad(rotation)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    X_rot = cos_r * (X - cx) - sin_r * (Y - cy) + cx
    Y_rot = sin_r * (X - cx) + cos_r * (Y - cy) + cy
    
    if pattern_id == 0:  # Spiral
        theta = dist * 0.1 + angle * 8
        pattern = np.sin(theta) * 128 + 127
    elif pattern_id == 1:  # Radial rings
        pattern = np.sin(dist * 0.08) * 128 + 127
    else:  # Grid waves
        pattern = np.sin(X_rot * 0.06 + Y_rot * 0.04) * 128 + 127
    
    return pattern

def select_source(img, seed):
    h, w = img.shape[:2]
    random.seed(seed)
    attempts = 0
    while attempts < 30:
        size = random.randint(80, max(h, w) // 3)
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        rect = img[y:y+size, x:x+size]
        if np.std(rect, axis=(0,1)).mean() > 40:
            resized = Image.fromarray(rect.astype(np.uint8)).resize((w, h), Image.LANCZOS)
            return np.array(resized, dtype=float), False, None
        attempts += 1
    
    pattern_id = random.randint(0, 2)
    rotation = random.uniform(0, 360)
    fallback = generate_fallback_pattern(h, w, pattern_id, rotation)
    print(f"[TrustLayer] Used fallback pattern {pattern_id} (rotated {rotation:.0f}°)")
    return fallback, True, (pattern_id, rotation)

def encode(image_path, message, output_path):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=float)
    h, w, _ = img.shape
    
    seed = int(hashlib.sha256(message.encode() if message else b'default').hexdigest(), 16) % (2**32)
    source, is_fallback, _ = select_source(img, seed)
    
    random.seed(seed)
    skip = random.choice([3, 5, 7])
    mask = np.zeros((h, w), dtype=bool)
    for i in range(h):
        for j in range(w):
            if hash((i, j, seed)) % skip == 0:
                mask[i, j] = True
    
    watermarked = img.copy()
    for ch in range(3):
        orig_f = fftshift(fft2(img[:,:,ch]))
        source_f = fftshift(fft2(source[:,:,ch]))
        
        dist = np.sqrt((np.ogrid[:h,:w][0] - h//2)**2 + (np.ogrid[:h,:w][1] - w//2)**2)
        high_mask = dist > (min(h, w) / 2 * 0.45)
        
        orig_f[high_mask] = orig_f[high_mask] * 0.2 + source_f[high_mask] * 0.8
        orig_f[mask & high_mask] += source_f[mask & high_mask] * 0.5
        
        watermarked[:,:,ch] = np.real(ifft2(ifftshift(orig_f)))
    
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    Image.fromarray(watermarked).save(output_path)
    print(f"[TrustLayer] Stamped saved: {output_path} ({'fallback' if is_fallback else 'self-similar'})")

def decode(image_path):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=float)
    h, w, _ = img.shape
    
    extracted = np.zeros((h, w, 3), dtype=float)
    for ch in range(3):
        f = fftshift(fft2(img[:,:,ch]))
        dist = np.sqrt((np.ogrid[:h,:w][0] - h//2)**2 + (np.ogrid[:h,:w][1] - w//2)**2)
        high_mask = dist > (min(h, w) / 2 * 0.45)
        high_f = f.copy()
        high_f[~high_mask] = 0
        ext_ch = np.real(ifft2(ifftshift(high_f)))
        ext_ch = (ext_ch - ext_ch.min()) / (ext_ch.ptp() + 1e-8)
        extracted[:,:,ch] = ext_ch * 255 * 6
    
    extracted = np.clip(extracted, 0, 255).astype(np.uint8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.imshow(img.astype(np.uint8))
    ax1.set_title("Original Image (Stamp Invisible)")
    ax1.axis('off')
    
    ax2.imshow(extracted)
    ax2.set_title("TrustLayer Reveal\n(Structured Pattern = AI-Labeled)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ["encode", "decode"]:
        print("Usage: python trustlayer.py encode <input> <output> [message_seed]")
        print("       python trustlayer.py decode <image>")
        sys.exit(1)
    
    if sys.argv[1] == "encode":
        message = sys.argv[4] if len(sys.argv) > 4 else ""
        encode(sys.argv[2], message, sys.argv[3])
    else:
        decode(sys.argv[2])