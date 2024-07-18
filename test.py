import spynet
import torchvision.transforms as T
from PIL import Image
import torch

tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[.485, .406, .456], 
                std= [.229, .225, .224])
])

model = spynet.SpyNet.from_pretrained('models/final.pt')
model.eval()

# frame1 = torch.randn(1, 3, 256, 256)
# frame2 = torch.randn(1, 3, 256, 256)
frame1 = tfms(Image.open('images/red_square.jpg').convert('RGB')).unsqueeze(0)
frame2 = tfms(Image.open('images/red_square2.jpg').convert('RGB')).unsqueeze(0)

flow = model((frame1, frame2))[0]
flow = spynet.flow.flow_to_image(flow)
Image.fromarray(flow).show()