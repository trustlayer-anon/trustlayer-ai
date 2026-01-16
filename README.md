# TrustLayer – Invisible AI Provenance Stamp

**Brought to you by the mysterious Mictoshi Nakamoto**  
*(the shadowy figure who allegedly dropped this code in a dark alley and vanished like a true provenance ghost — voluntary labeling only, no double-spending allowed)*

Open-source invisible watermark for AI images and audio – voluntary transparency.

**Images** – high-frequency self-similar stamp  
**Audio** – inaudible ultrasonic + infrasound bands

Local, offline, MIT licensed.

Usage:
```bash
# Image
python trustlayer.py encode input.jpg output.jpg "seed"
python trustlayer.py decode output.jpg

# Audio
python trustlayer_audio.py encode input.wav output.wav "seed"
python trustlayer_audio.py decode output.wav  # → reveal.wav
