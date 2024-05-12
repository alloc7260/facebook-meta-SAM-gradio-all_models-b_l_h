import gradio as gr
from PIL import Image
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

inimg = gr.Image()
outimg = gr.AnnotatedImage()

custom_html = '''
<center>
  <div style="overflow:hidden ; max-width: fit-content; margin-left: auto; margin-right: auto;">
      <div style="float: left; font-family: Arial; font-size: 25px; color: orange; margin: 10px;">Please</div>
        <div style="float: left">
            <a href="https://www.buymeacoffee.com/alloc7260">
                <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=☕&slug=alloc7260&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" />
            </a>
        </div>
      <div style="float: left; font-family: Arial; font-size: 25px; color: orange; margin: 10px;">for upgrade to gpu instance</div>
  </div>
</center>
'''

model_type = ["vit_h", "vit_l", "vit_b"]
model_path = {
    "vit_h": "weights/sam_vit_h_4b8939.pth",
    "vit_l": "weights/sam_vit_l_0b3195.pth",
    "vit_b": "weights/sam_vit_b_01ec64.pth"
}

area_threshold = 1 # in percentage

def resize_image(image, max_width=1500):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_width = min(original_width, max_width)
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def gen_mask(model_type, img):
    print(model_type)

    numpy_array = np.array(img)
    img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    sam = sam_model_registry[model_type](checkpoint=model_path[model_type]).to(device='cuda:0')
    tot_pix = img.shape[0] * img.shape[1]
    mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=tot_pix*(area_threshold/100))

    with torch.inference_mode():
        masks = mask_generator.generate(img)
        torch.cuda.empty_cache()

    filtered_masks = [mask for mask in masks if mask['area'] > tot_pix*(area_threshold/100)]

    segmentations = [
        mask['segmentation']
        for mask
        in sorted(filtered_masks, key=lambda x: x['area'], reverse=True)
    ]
    return segmentations

def mask(img, model_type):
    PIL_image = Image.fromarray(img.astype('uint8'), 'RGB')
    resimg = resize_image(PIL_image)
    outputs_masks = gen_mask(model_type, resimg)
    return (resimg, [(outputs_masks[i], str(i + 1)) for i in range(len(outputs_masks))])

interface = gr.Interface(mask, 
                        inputs=[inimg, 
                            gr.Dropdown(model_type, label="Model Type")
                        ], 
                        outputs=outimg,
            )

with interface:
  che = gr.HTML(custom_html)

interface.launch(show_api=False, debug=True, server_name="0.0.0.0", server_port=1156)  