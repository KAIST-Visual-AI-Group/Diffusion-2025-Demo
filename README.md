<div align=center>
  <h1>
    Demo Session
  </h1>
  <p>
    <a href=https://diffusion.kaist.ac.kr/ target="_blank"><b>KAIST CS492(C): Diffusion and Flow Models (Fall 2025)</b></a><br>    
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://dvelopery0115.github.io/ target="_blank"><b>Seungwoo Yoo</b></a>  (dreamy1534 [at] kaist.ac.kr)<br>
  </p>
</div>

<div align=center>
  <img src="./media/teaser.png" width="768"/>
</div>

---

## Description
In this demo session, you will gain hands-on experience with Stable Diffusion 3—one of the most powerful text-to-image models—while experimenting with three popular techniques: [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598), [ControlNet](https://arxiv.org/abs/2302.05543), and [LoRA](https://arxiv.org/abs/2106.09685).

(1) **Classifier-Free Guidance** improves the effect of user-provided conditional inputs, such as text prompts, on the generated images. In this demo, you will explore how changing the guidance scale influences the outputs.

(2) **ControlNet** further extends the capabilities of text-to-image diffusion models, such as Stable Diffusion, by allowing them to incorporate additional conditions beyond text prompts, such as sketches or depth maps. In this demo, we will present a minimal example of using ControlNet to generate images that follow depth maps.

(3) **LoRA (Low-Rank Adaptation)** is an efficient fine-tuning technique for neural networks that enables the customization of diffusion models with relatively small datasets, ranging from a few images to a few thousand. This repository provides a minimal example, adapted from [a LoRA implementation for Stable Diffusion 3.5](https://github.com/seochan99/stable-diffusion-3.5-text2image-lora), to support future projects in visual content creation.

This material is heavily based on the [diffusers](https://github.com/huggingface/diffusers) library. You are strongly encouraged to consult materials beyond the scope of this demo, as they will be valuable for your projects.

## Setup
Install the required package within the `requirements.txt`.

**NOTE:** Install PyTorch according to the CUDA version of your environment (See [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/))
```
conda create -n cs492c python=3.10
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Before proceeding with the demo, we will have a look at [Hugging Face](https://huggingface.co/), an open-source platform that serves as a hub for machine learning applications. [Diffusers](https://github.com/huggingface/diffusers) a go-to library for pretrained diffusion models made by Hugging Face. As we'll be downloading the pretrained Stable Diffusion model from Hugging Face, you'll need to ensure you have access tokens.

Before running the demo, please do the following:
* Sign into Hugging Face.
* Obtain your Access Token at `https://huggingface.co/settings/tokens`.
* In your terminal, log into Hugging Face by `$ huggingface-cli login` and enter your Access Token.

You can check whether you have access to Hugging Face using the below code, which downloads Stable Diffusion from Hugging Face and generates an image with it.

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```

All set! Please open up the Jupyter Notebook named `demo_cfg_controlnet.ipynb` to get started!

## ❗Ethical Usage
While you are encouraged to explore creative possibilities using the above methods, it is crucial that you do not use these personalization techniques for **harmful purposes**, such as generating content that includes nudity, violence, or targets specific identities. It is your responsibility to ensure that this method is applied ethically.

## Credits
This repository is built primarily on the [Diffusers](https://huggingface.co/docs/diffusers/index) library. We also thank the authors of the following resources:
- [LoRA Implementation for Stable Diffusion 3.5](https://github.com/seochan99/stable-diffusion-3.5-text2image-lora)
- [A Dataset of Rubber Duck Images](https://huggingface.co/datasets/linoyts/rubber_ducks)

## Further Readings
If you are interested in this topic, we encourage you to check out the materials below.

* [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
* [ControlNet Github](https://github.com/lllyasviel/ControlNet)
* [ControlNet Hugging Face Documentation](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
* [GLIGEN: Open-Set Grounded Text-to-Image Generation](https://arxiv.org/abs/2301.07093)
* [Style Aligned Image Generation via Shared Attention](https://arxiv.org/abs/2312.02133)
* [StyleDrop: Text-to-Image Generation in Any Style](https://arxiv.org/abs/2306.00983)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning](https://github.com/cloneofsimo/lora)
* [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
* [Multi-Concept Customization of Text-to-Image Diffusion](https://arxiv.org/abs/2212.04488)
