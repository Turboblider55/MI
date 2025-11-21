# Működés: generál egy rövid (5-6 mondatos) történetet, feldarabolja mondatokra (max N),
# majd minden mondathoz 512x512 képet generál Stable Diffusion v1-5-tel, és BLIP-pel képleírást ad.
#
# Telepítés (példa CUDA 11.8 / Windows/Linux):
# pip install -U "torch" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers accelerate safetensors pillow

import os
import time
import textwrap
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
from PIL import Image

# -------------------------
# Script könyvtár (output ide kerül)
# -------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR / "output"
IMG_DIR = OUTPUT_DIR / "images"
CAP_DIR = OUTPUT_DIR / "captions"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CAP_DIR, exist_ok=True)

# -------------------------
# Konfiguráció
# -------------------------

FLAN_MODEL_ID = "google/flan-t5-xl"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CAP_MODEL_ID = "Salesforce/blip-image-captioning-base"

STORY_PROMPT = (
    "Write a story in 5-6 sentences about a lonely dragon who wants to befriend humans. "
    "Keep each sentence as a clear visual scene that could be illustrated."
)

IMAGE_SIZE = (512, 512)
SD_INFERENCE_STEPS = 25

# segédfunc
def clear_gpu():
    """Törli a CUDA cache-t (ha van) és futtat GC-t."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    time.sleep(0.2)

# -------------------------
# 1) Történet generálása (FLAN-T5-XL) - CPU
# -------------------------

def generate_story_cpu(prompt: str) -> str:
    print("1) Történetgenerálás: betöltés FLAN-T5-XL (CPU). Ez néhány másodpercet/percet is igénybe vehet RAM-tól függően.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_ID)  # CPU default
    except Exception as e:
        print("Hiba: nem sikerült betölteni a Flan-T5-XL modellt. Hiba:", e)
        return ""

    story_text = ""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        story_text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # mentés
        story_path = OUTPUT_DIR / "story.txt"
        story_path.write_text(story_text, encoding="utf-8")
        print(f" -> Generált történet elmentve: {story_path}")
        print(textwrap.fill(story_text, width=100))
    except Exception as e:
        print("Hiba a történet generálása közben:", e)
    finally:
        # memóriafelszabadítás - fontos hogy CPU-s modell után GPU tiszta legyen
        try:
            del tokenizer, model
        except Exception:
            pass
        clear_gpu()
    return story_text

# -------------------------
# 2) Mondatokra vágás (nem kényszerítve)
# -------------------------

def split_into_sentences(text: str):
    import re
    if not text:
        return []
    # egyszerű, megbízható felbontás: .!? után
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [p.strip() for p in parts if len(p.strip()) > 3]

    return sentences

# -------------------------
# 3) Stable Diffusion betöltése (SD v1-5) és képgenerálás
# -------------------------

def load_sd(sd_model_id: str):
    print(f"{sd_model_id} betöltése (FP32, CUDA)...")

    gc.collect()
    torch.cuda.empty_cache()

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    # Biztosítani, hogy minden modul FP32 és GPU-s
    pipe.unet.to(device="cuda", dtype=torch.float32)
    pipe.vae.to(device="cuda", dtype=torch.float32)
    pipe.text_encoder.to(device="cuda", dtype=torch.float32)

    print("Betöltve a GPU-ra FP32 módban!")
    return pipe

def generate_images(pipe, prompts, image_size=IMAGE_SIZE, steps=SD_INFERENCE_STEPS):
    image_paths = []
    if pipe is None:
        print("SD pipeline None - képgenerálás kihagyva.")
        return image_paths

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n3) Kép generálás {i}/{len(prompts)} ...")
        try:
            out = pipe(prompt, height=image_size[0], width=image_size[1], num_inference_steps=steps)
            img = out.images[0]
            path = IMG_DIR / f"part_{i}.png"
            img.save(path)
            image_paths.append(str(path))
            print(" -> Kép mentve:", path)
        except torch.cuda.OutOfMemoryError as oom:
            print("OOM hiba a képgenerálásnál (512x512).", oom)
            # Tovább haladunk a következő prompttal (nem próbálunk kisebbet)
        except Exception as e:
            print("Hiba a képgenerálásnál:", e)

    # SD után töröljük a pipe-ot és felszabadítjuk a memóriát
    try:
        # ha vannak nagy komponensek, töröljük
        if hasattr(pipe, "unet"):
            try:
                del pipe.unet
            except Exception:
                pass
        if hasattr(pipe, "vae"):
            try:
                del pipe.vae
            except Exception:
                pass
        if hasattr(pipe, "text_encoder"):
            try:
                del pipe.text_encoder
            except Exception:
                pass
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    
    clear_gpu()
    return image_paths

# -------------------------
# 4) BLIP képaláírás (GPU preferált, de CPU fallback)
# -------------------------

def caption_images(image_paths):
    if not image_paths:
        print("Nincsenek képek captionolásra.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"4) BLIP betöltése a(z) {device} eszközre ...")
    try:
        processor = BlipProcessor.from_pretrained(CAP_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(
            CAP_MODEL_ID,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": "cuda"},
            use_safetensors=True,
            local_files_only=True,
            use_fast=True
            ).to(device)
    except Exception as e:
        print("Hiba a BLIP modell betöltésénél:", e)
        return

    for i, img_path in enumerate(image_paths, start=1):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=60)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            # mentés: prompt+caption
            cap_file = CAP_DIR / f"part_{i}.txt"
            content = f"Image file: {img_path}\n\nImage caption:\n{caption}\n"
            cap_file.write_text(content, encoding="utf-8")
            print(f" -> Caption mentve: {cap_file}")
        except Exception as e:
            print("Hiba caption generálásnál:", e)

    # BLIP után törlés és tisztítás
    try:
        del processor, model
    except Exception:
        pass
    clear_gpu()

# -------------------------
# Fő futtatás
# -------------------------

def main():
    print("=== Pipeline start ===")
    # 1) story (FLAN T5 XL on CPU)
    story = generate_story_cpu(STORY_PROMPT)
    if not story:
        print("Történetgenerálás nem sikerült — kilépés.")
        return

    # 2) split into sentences
    sentences = split_into_sentences(story)
    if not sentences:
        print("Nincsenek mondatok a generált történetben — kilépés.")
        return

    print("\nMondatok, amiket illusztrálunk (maximum):")
    for idx, s in enumerate(sentences, start=1):
        print(f"{idx}. {s}")

    # egyes mondatokat mentünk prompt fájlokba
    for idx, s in enumerate(sentences, start=1):
        pfile = OUTPUT_DIR / f"part_{idx}_prompt.txt"
        pfile.write_text(s, encoding="utf-8")

    # 3) load SD and generate images
    sd_pipe = load_sd(SD_MODEL_ID)
    if sd_pipe is None:
        print("Stable Diffusion betöltése nem sikerült. Kilépés.")
        return

    image_paths = generate_images(sd_pipe, sentences, image_size=IMAGE_SIZE, steps=SD_INFERENCE_STEPS)
    if not image_paths:
        print("Nem készült kép. Kilépés.")
        return

    # 4) caption images with BLIP
    caption_images(image_paths)

    print("\n=== Pipeline complete ===")
    print("Output mappa:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
