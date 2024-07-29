import Live_Detection.model as md
import Live_Detection.transform as trans
import torch
import numpy as np
import torch.nn.functional as F
def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size

def Live_Detect(img,model_path,input_size=[80,80]):
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')
    model=md.MiniFASNetV2(conv6_kernel=get_kernel(80,80)).to(device)
    # load state_dict
    state_dict = torch.load(model_path, map_location=device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            new_state_dict[name_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    # predict
    test_transform = trans.Compose([
            trans.ToTensor(),
        ])
    img = test_transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        result = model.forward(img)
        result = F.softmax(result,dim=1).cpu().numpy()
    label = np.argmax(result)
    # label==1 -> Live_Face
    return int(label)