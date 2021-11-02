import torch
from torch.utils.data import DataLoader
import pre_processing as pp
import config as cg
import main

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_data = pp.classify_data(txt=cg.data_dir + '/testing_img_order.txt')
test_data = main.LoadDataset(all_data, transform=pp.image_transforms['valid'])
test_data_loader = DataLoader(test_data, batch_size=cg.batch_size, shuffle=False)
model = main.resnext50.to(device)
torch.load('models/{}.pt'.format(cg.epochs))

predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, img in enumerate(test_data_loader):
        inputs = img
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y + 1)

with open('./birds/testing_img_order.txt', 'r') as file:
    answer = open('answer.txt', 'w')
    label = []
    for y in predict:
        label.append(pp.find_class_name(y))

    i = 0
    for line in file:
        line = line.strip('\n')
        print(line)
        answer.write(line + ' ' + label[i] + '\n')
        i = i + 1
