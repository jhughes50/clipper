"""
    @Author Jason Hughes
    @Date December 2025

    @About jit compile the model componenets
"""
import cv2
import torch
import numpy as np
from PIL import Image 
import torch.nn.functional as F
from processing_clipseg import CLIPSegProcessor



processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

# compile the image model

from modeling_clipseg import CLIPSegVisionTransformer, CLIPSegModel, CLIPSegDecoder, CLIPSegForImageSegmentation


model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined").to('cuda')
#print(model.vision_model.to('cuda'))
image = np.array(Image.open("test4.png"))[:,:,:-1]
image = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)
H,W,C = image.shape

inputs = processor(text=["road"], images=image, return_tensors="pt", padding="max_length",max_length=10)

img_output = model.vision_model(pixel_values=inputs["pixel_values"].to('cuda'), 
                            output_attentions=False, 
                            output_hidden_states=True, 
                            interpolate_pos_encoding=True,
                            return_dict=False)

text_output = model.text_model(input_ids=inputs["input_ids"].to('cuda'), 
                               attention_mask=inputs["attention_mask"].to('cuda'), 
                               return_dict=False)



#traced_model = torch.jit.trace(model.vision_model, inputs["pixel_values"])
#torch.jit.save(traced_model, "clip-vision-model-traced.pt")

#traced_model = torch.jit.trace(model.visual_projection, output[1])
#torch.jit.save(traced_model, "clip-vision-projection-traced.pt")

#traced_model = torch.jit.trace(model.text_model, (inputs["input_ids"].to('cuda'), inputs["attention_mask"].to('cuda')))
#torch.jit.save(traced_model, "clip-text-model-traced.pt")

#output = model.text_model(inputs["input_ids"].to('cuda'), inputs["attention_mask"].to('cuda'))
#
#traced_model = torch.jit.trace(model.text_projection,  output[1])
#torch.jit.save(traced_model, "clip-text-projection-traced.pt")

decoder = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to('cuda')
img_projected = model.visual_projection(img_output[1])
txt_projected = model.text_projection(text_output[1])
activations = [img_output[2][i + 1] for i in decoder.extract_layers]

print(decoder.extract_layers)
print(img_projected.shape)
print(type(txt_projected))
print(type(activations))
print(type(activations[1]))
#traced_model = torch.jit.trace(decoder.decoder, (activations, txt_projected))
#torch.jit.save(traced_model, "clip-decoder-traced.pt")
output = decoder.decoder(activations, txt_projected, return_dict=False)
print(output[0].shape)

heatmap_p = output[0].squeeze()

heatmap = F.interpolate(heatmap_p.unsqueeze(0).unsqueeze(0),
                        size=(H,W),  # Or your original image size
                        mode='bilinear',
                        align_corners=False).squeeze()

heatmap_np = heatmap.squeeze().cpu().detach().numpy()

heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
import matplotlib.pyplot as plt
plt.imshow(heatmap_np, cmap='hot')
plt.show()
