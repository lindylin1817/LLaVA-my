import gradio as gr

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

import requests
from PIL import Image
from io import BytesIO
import re

model_path = "/home/yhlin/models/llava-v1.5-7b-task"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "./images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
#    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "model": None,
    "tokenizer": None,
    "image_processor": None,
    "context_len": None
})()

def image_parser(image_file, args):
    out = image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image
def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def init_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    args.conv_mode = conv_mode
    args.model = model
    args.tokenizer = tokenizer
    args.image_processor = image_processor
    args.context_len = context_len


def eval_model(input_image1, input_image2, query):
    qs = query
    if input_image1 is None:
        input_images = [input_image2]
    elif input_image2 is None:
        input_images = [input_image1]
    else:
        input_images = [input_image1, input_image2]
    num_images = len(input_images)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    print(image_token_se)
    if IMAGE_PLACEHOLDER in qs:
        if args.model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if args.model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            image_token_str = ""
            for i in range(num_images):
                image_token_str = image_token_str + DEFAULT_IMAGE_TOKEN
            qs = image_token_str + "\n" + qs
#            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

#    image_files = image_parser(image_file, args)
#    images = load_images(image_files)

    images_tensor = process_images(
        input_images,
        args.image_processor,
        args.model.config
    ).to(args.model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, args.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, args.tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = args.model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = args.tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs

if __name__ == "__main__":
    init_model(args)
    iface = gr.Interface(fn=eval_model,
                         inputs=[gr.inputs.Image(label="Input image 1", shape=(336, 336), type="pil"),
                                 gr.inputs.Image(label="Input image 2", shape=(336, 336), type="pil"),
                                gr.inputs.Textbox(label="Input text 1", type="text")],
                         outputs="text")
#    eval_model(args, prompt, image_file)
    iface.launch(server_name="0.0.0.0")


