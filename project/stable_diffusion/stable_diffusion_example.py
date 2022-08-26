# ---
# jupyter:
#   jupytext:
#     formats: '@/ipynb,docs//md,py:percent'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.8.10 64-bit
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stable Diffusion with diffusers
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb

# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
import diffusers
import torch
import os


def load_img2img(
    reponame: str = "CompVis/stable-diffusion-v1-4",
    revision: str = "fp16",
    torch_dtype: type = torch.float16,
):
    with open("/home/vscode/.huggingface/token", "r") as file:
        token = file.read(-1)

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        reponame,
        revision=revision,
        torch_dtype=torch_dtype,
        use_auth_token=token,
    )

    return pipe


img2img = load_img2img().to("cuda")

# %%
from torch import autocast, Generator
from typing import List
from PIL import Image

def gen_txt2img(prompt:List[str], seed=-1):
    global img2img

    if seed == -1:
        with autocast("cuda"):
            result = img2img(prompt)
            return result["sample"]
    else:
        gen = Generator("cuda").manual_seed(1024)
        with autocast("cuda"):
            result = img2img(prompt, num_inference_steps=15, generator=gen)
            return result["sample"]

def grid_img(imgs, rows:int, cols:int):
    if len(imgs) != rows * cols:
        raise Exception("rows * cols != len(imgs)")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    
    return grid


# %%
imgs = gen_txt2img([
    "melancholic wallpaper, granblue fantasy, japanese 15 years old girl, crying and dancing alone, putting on a hood, shining black eyes, background is midnight forest"
] * 4)

grid_img(imgs=imgs, rows=2, cols=2)

# %%
# import torch
# from typing import Tuple
# from tqdm.auto import tqdm
# from torch import autocast
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler


# class StableDiffusion:

#     def init(self,
#              sd_reponame: str = "CompVis/stable-diffusion-v1-4",
#              txtencoder_reponame: str = "openai/clip-vit-large-patch14",
#              revision: str = "fp16",
#              torch_dtype: type = torch.float16,
#              manual_seed: int = -1,
#              device="cuda"):

#         with open(os.getenv("HOME")) as file:
#             token = file.read(-1)

#         self.vae = AutoencoderKL.from_pretrained(
#             sd_reponame,
#             subfolader="vae",
#             revision=revision,
#             torch_dtype=torch_dtype,
#             use_auth_token=token,
#         ).to(device)

#         self.tokenizer = CLIPTokenizer.from_pretrained(txtencoder_reponame).to(
#             device)
#         self.txtencoder = CLIPTextModel.from_pretrained(txtencoder_reponame).to(
#             device)

#         self.unet = UNet2DConditionModel.from_pretrained(
#             sd_reponame,
#             subfolder="unet",
#             revision=revision,
#             torch_dtype=torch_dtype,
#             use_auth_token=token,
#         ).to(device)

#         self.generator = Generator(device)
#         if manual_seed >= 0:
#             self.generator = self.generator.manual_seed(seed=manual_seed)

#         self.device = device
#         self.beta_start = 0.00085
#         self.beta_end = 0.012

#     def txt2img(
#         self,
#         prompt: str,
#         size: Tuple[int, int] = (512, 512),
#         num_inference_steps: int = 100,
#         guidance_scale=7.5,
#     ):
#         width, height = size
#         scheduler = LMSDiscreteScheduler(
#             beta_start=self.beta_start,
#             beta_end=self.beta_end,
#             beta_schedule="scaled_linear",
#             num_train_timestamps=1000,
#         )

#         # 文章からEmbeddingsに変換する
#         token = self.tokenizer(
#             [prompt],
#             padding="max_length",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt",
#         )

#         with torch.no_grad():
#             txt_embeddings = self.txtencoder(token.input_ids.to(self.device))[0]

#         max_length = token.input_ids.shape[-1]
#         empty_token = self.tokenizer(
#             [""],
#             padding="max_length",
#             max_length=max_length,
#             return_tensors="pt",
#         )

#         with torch.no_grad():
#             empty_txt_embeddings = self.txtencoder(
#                 empty_token.input_ids.to(self.device))[0]

#         embeddings = torch.cat([empty_txt_embeddings, txt_embeddings])

#         # ランダムノイズを生成
#         torch.randn(
#             size=(1, self.unet.in_channels, height // 8, width // 8),
#             generator=self.generator,
#         ).to(device=self.device)

#         scheduler.set_timesteps(num_inference_steps)


# %%
# from transformers import CLIPTokenizer
# from diffusers import AutoencoderKL, UNet2DConditionModel
# import os

# with open(os.getenv("HOME", "")+"/.huggingface/token", "r") as file:
#     token = file.read(-1)

# UNet2DConditionModel.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     subfolder="unet",
#     revision="fp16",
#     torch_dtype=torch.float16,
#     use_auth_token=token,
# )

