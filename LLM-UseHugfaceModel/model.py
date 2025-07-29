# 本项目使用来自HuggingFace已预训练好的模型

from diffusers import AutoPipelineForText2Image
from PIL import Image
import time
import torch

start_time = time.time()
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save('test1.png')

end_time = time.time()

gap_time = end_time - start_time
print("script execution time:",gap_time," seconds")
