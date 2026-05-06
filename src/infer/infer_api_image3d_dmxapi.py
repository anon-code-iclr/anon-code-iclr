import argparse
import io
import json
import os
from PIL import Image
from tqdm import tqdm

import nibabel as nib
import numpy as np
from google import genai
from google.genai import types


args = argparse.ArgumentParser()
args.add_argument("--input_path", type=str, default=None)
args.add_argument("--output_dir", type=str, default=None)
args.add_argument("--model", type=str, default=None)
args.add_argument("--base_url", type=str, default=None)
args.add_argument("--api_key", type=str, default=None)
args.add_argument("--num_slices", type=int, default=3)
args.add_argument("--window_center", type=float, default=None)
args.add_argument("--window_width", type=float, default=None)
args = args.parse_args()

client = genai.Client(
    api_key=args.api_key,
    http_options={"base_url": args.base_url} if args.base_url else None,
)
system_prompt=""


def pil_to_part(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=buffer.getvalue(),
        ),
        media_resolution={"level": "media_resolution_medium"},
    )


def build_user_content(
    item,
    num_slices=3,
    window_center=None,
    window_width=None,
):
    """
    Build Gemini multimodal content:
    [image_part, image_part, "..."]
    """
    content = []

    for nii_path in item["videos"]:
        if not os.path.exists(nii_path):
            print(f"[Warning] NII file not found: {nii_path}")
            continue

        nii = nib.load(nii_path)
        image3d = nii.get_fdata()  # (X, Y, Z) == (W, H, T)

        if window_center is not None and window_width is not None:
            hu_min = window_center - window_width // 2
            hu_max = window_center + window_width // 2
            image3d = np.clip(image3d, hu_min, hu_max)
            image3d = (image3d - hu_min) / (hu_max - hu_min) * 255

        depth = image3d.shape[2]
        # indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        indices = np.linspace(0, depth - 1, num_slices + 2, dtype=int)[1:-1]

        slices = []
        for idx in indices:
            slice = image3d[:, :, idx]
            img = Image.fromarray(slice).convert("RGB")
            slices.append(img)

        for slice in slices:
            content.append(pil_to_part(slice))

    prompt = item["messages"][-2]["content"]
    content.append(types.Part(text=prompt.replace("<video>", "").strip()))
    return content


if __name__ == "__main__":
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, item in tqdm(enumerate(data), total=len(data)):
        if os.path.exists(os.path.join(args.output_dir, f"{idx}.json")):
            continue

        try:
            user_content = build_user_content(
                item,
                num_slices=args.num_slices,
                window_center=args.window_center,
                window_width=args.window_width
            )
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    types.Content(
                        parts=([types.Part(text=system_prompt)] if system_prompt else []) + user_content,
                    )
                ],
                # temperature=0.2,
                # max_tokens=1024,
            )
            output_text = response.text
        except Exception as e:
            print(f"[Error] Failed to get response for item {item['id']}: {e}")
            continue

        # item["messages"][-1]["content"] = output_text
        item["predict"] = output_text
        print(f"Predict:\n{output_text}\n")

        with open(os.path.join(args.output_dir, f"{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=4)

        # if idx > 3:
        #     break
