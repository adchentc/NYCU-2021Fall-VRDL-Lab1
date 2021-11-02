from torchvision import transforms


def image_transforms():
    trans = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return trans


def classify_data(txt):
    file = open(txt, 'r')
    data = []
    for line in file:
        line = line.strip('\n')
        line = line.split(' ')
        if len(line) != 1:
            labels = line[1].split('.')
            data.append((line[0], labels[0]))  # list include tuple [(), (), ()]
        else:
            data.append(line[0])
    return data


def find_class_name(label):
    result = ''
    with open('./birds/classes.txt', 'r') as file:
      for line in file:
        whole_class = line.strip('\n')
        label_num = whole_class.split('.')[0]
        if label == int(label_num):
          result = whole_class
    return result
