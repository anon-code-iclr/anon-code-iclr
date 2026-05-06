import argparse
import json
import mimetypes
import os
from tqdm import tqdm

from google import genai
from google.genai import types


args = argparse.ArgumentParser()
args.add_argument("--input_path", type=str, default=None)
args.add_argument("--output_dir", type=str, default=None)
args.add_argument("--model", type=str, default=None)
args.add_argument("--base_url", type=str, default=None)
args.add_argument("--api_key", type=str, default=None)
args = args.parse_args()

client = genai.Client(
    api_key=args.api_key,
    http_options={"base_url": args.base_url} if args.base_url else None,
)
system_prompt=""


def read_image_to_part(image_path: str):
    """
    Read a local image file and convert it to a Gemini image part.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    return types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=image_bytes,
        ),
        media_resolution={"level": "media_resolution_medium"},
    )


def build_user_content(item):
    """
    Build Gemini multimodal content:
    [image_part, image_part, "..."]
    """
    content = []

    for image_path in item["images"]:
        if len(content) >= 3:
            break

        if not os.path.exists(image_path):
            print(f"[Warning] Image not found: {image_path}")
            continue

        content.append(read_image_to_part(image_path))

    prompt = item["messages"][-2]["content"]
    content.append(types.Part(text=prompt.replace("<image>", "").strip()))
    return content


if __name__ == "__main__":
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, item in tqdm(enumerate(data), total=len(data)):
        if os.path.exists(os.path.join(args.output_dir, f"{idx}.json")):
            continue

        try:
            user_content = build_user_content(item)
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
