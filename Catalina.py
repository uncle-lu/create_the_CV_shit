# created by uncle-lu
# final car line shit
# 2021.06.01

import warnings
import numpy as np
from torch.autograd import Variable
from albumentations.core.composition import Transforms
from dataloader import *
from model.model import LaneNet, compute_loss
from average_meter import *
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_folder = "./seg_result"
dataset = './test'

def getkey(x):
    return abs(x-256)

def turnto1952(x):
    return int(x*1952/512)

def merge_array(line):
    if len(line) == 0:
        return 0, 0
    this_ans = []
    l = line[0]
    r = line[0]-1
    for i in line:
        if i != r+1:
            this_ans.append((l+r)/2)
            l = i
        r = i
    this_ans.append((l+r)/2)
    this_ans.sort(key = getkey)
    return this_ans[0] if len(this_ans) > 0 else 0, this_ans[1] if len(this_ans) >= 2 else 0

def img_get_line(img, y):
    sit = -1
    points = []
    for (r,g,b) in img[y,:]:
        sit = sit + 1
        if r == 0 and g == 0 and b == 0:
            continue
        points.append(sit)
    return points

def Traversal(img):
    ans1, ans2 = merge_array(img_get_line(img, 191))
    ans3, ans4 = merge_array(img_get_line(img, 255))
    ans1 = turnto1952(ans1)
    ans2 = turnto1952(ans2)
    ans3 = turnto1952(ans3)
    ans4 = turnto1952(ans4)
    return ans1, ans2, ans3, ans4

if __name__ == '__main__':
	val_dataset_file = os.path.join(dataset, 'train.txt')
	val_dataset = LaneDataSet(val_dataset_file, stage = 'val')
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
	model = torch.load('./save_pth/1000.pth', map_location=DEVICE)
	model.eval()
	for batch_idx, (image_data, binary_label, instance_label) in enumerate(val_loader):
		image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
		with torch.set_grad_enabled(False):
                    net_output = model(image_data)
                    seg_logits = net_output["seg_logits"].cpu().numpy()[0]
                    result = (np.argmax(seg_logits, axis=0)*127).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_folder, '{0:04d}-b.png'.format(batch_idx)), result)
                    img = cv2.imread(os.path.join(save_folder,'{0:04d}-b.png'.format(batch_idx)))
                    ans1, ans2, ans3, ans4 = Traversal(img)
                    file_path = os.path.join(dataset,"result","{0:04d}.txt".format(batch_idx))
                    resualt_file = open(file_path,"a")
                    resualt_file.write("{} {}\n".format(ans1, 810))
                    resualt_file.write("{} {}\n".format(ans2, 810))
                    resualt_file.write("{} {}\n".format(ans3, 1080))
                    resualt_file.write("{} {}\n".format(ans4, 1080))
                    resualt_file.close()