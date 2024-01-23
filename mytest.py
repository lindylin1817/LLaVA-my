from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "/home/yhlin/models/llava-v1.5-7b"
print(get_model_name_from_path(model_path))
#tokenizer, model, image_processor, context_len = load_pretrained_model(
#    model_path=model_path,
#    model_base=None,
#    model_name=get_model_name_from_path(model_path)
#)


prompt = "这里的两张图片之间的差异是有物体变化了？还是说差异只是光影变化？"
#prompt = "这里有多少张图片？"
image_file = "./images/nochange1_1.jpg,./images/nochange1_2.jpg"
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
image_files = image_parser(args)
print(image_files)
eval_model(args)
