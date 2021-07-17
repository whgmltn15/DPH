from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import torch


class NkDataSet(Dataset):

    # 데이터 초기화, init = 처음만들 때 설정 ToTensor = 0 에서 1사이로 값 바꾸기 iloc = 데이터 잘라서 가져오기
    def __init__(self, file_path):

        self.trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
        self.data_info = pd.read_csv(file_path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0][1:])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1][1:])
        self.label_arr = torch.from_numpy(self.label_arr)
        self.data_len = len(self.data_info.index)

    # 경로를 통해서 실제 데이터에 접근해서 데이터를 돌려주는 함수 image_arr = 이미지에 대한 경로
    def __getitem__(self, index):

        img_name = self.image_arr[index]
        img_as_img = Image.open(img_name)
        img_as_tensor = self.trans(img_as_img)
        img_label = self.label_arr[index]

        return img_as_tensor, img_label

    # 데이터의 전체 길이 구하는 함수
    def __len__(self):
        return self.data_len

# 파이토치가 받아들일 수 있는 것으로 데이터 로딩
def get_data_loader(args):
    csv_path = '.\\data\\test.csv' # train 에 대한 경로
    train_dataset = NkDataSet(csv_path)

    val_csv_path = '.\\data\\test.csv' # 평가하는 데이터셋의 경로
    val_dataset = NkDataSet(val_csv_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # batch_size = 학습할 때 몇개씩 묶을지

    return train_loader, val_loader

# csv 경로 설정
csv_path = '.\\data\\test.csv'
custom_dataset = NkDataSet(csv_path)
my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=1, shuffle=False)

# enumerate 는 list에 있는 내용을 순서 매기면서 프린트
for i, (images, labels) in enumerate(my_dataset_loader):
     print(labels, images.shape)