# Tokenize images to components for next component prediction

## Installation

```
conda env create -f titok.yml
pip install -r req.txt
pip install --upgrade diffusers[torch]
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Implementation of the tokenizer

The tokenizer is implemented in the `paintmind/stage1/diffuse_slot.py` file. The `DiffuseSlot` class is responsible for the tokenization of the images. 
This class will load a pretrained VAE from stable diffusion, and initialize a diffusion transformer(DiT) to learn the auto-encoding (tokenization) of the images.
A vision transformer is used to encode the image, and the tokens from that vision transformer are used as the input to a slot attention module which introduces the order to the tokens (the `CausalSemanticGrouping` class).
Then we use the nested attention which follows the implementation of the nested dropout to order the information learned by each of the slots (`NestedAttention` class).
The resulting tokens are used as conditions for the DiT to denoise the latent of the image to generate the image.
