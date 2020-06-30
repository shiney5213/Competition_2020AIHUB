import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
from torch import optim
from torch.autograd import Variable
import torchvision.utils

from dataloader import data_loader
from evaluation import evaluation_metrics
from model import SiameseNetwork
import torch.nn.functional as F

'''
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
** 컨테이너 내 기본 제공 폴더
- /datasets : read only 폴더 (각 태스크를 위한 데이터셋 제공)
- /tf/notebooks :  read/write 폴더 (참가자가 Wirte 용도로 사용할 폴더)
1. 참가자는 /datasets 폴더에 주어진 데이터셋을 적절한 폴더(/tf/notebooks) 내에 복사/압축해제 등을 진행한 뒤 사용해야합니다.
   예시> Jpyter Notebook 환경에서 압축 해제 예시 : !bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   예시> Terminal(Vs Code) 환경에서 압축 해제 예시 : bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   
2. 참가자는 각 문제별로 데이터를 로드하기 위해 적절한 path를 코드에 입력해야합니다. (main.py 참조)
3. 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (main.py 참조)
4. 세션/컨테이너 등 재시작시 위에 명시된 폴더(datasets, notebooks) 외에는 삭제될 수 있으니 
   참가자는 적절한 폴더에 Dataset, Source code, 결과 파일 등을 저장한 뒤 활용해야합니다.
   
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''


'''
# 수정한 부분
1.  reapeat num: 30
2. epoch: 30
3. lr: 0.00006
4. 모델 : 돈호님 참고모델
5. 모델 이름: base1_.pth
6. input: (1, 105, 105
)
'''

try:
    from nipa import nipa_data
    DATASET_PATH = nipa_data.get_data_root('deepfake')
except:
    DATASET_PATH = os.path.join('./data/')

def _infer(model, cuda, data_loader):
    res_id = []
    res_fc = []

    euclidean_distances = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            iter1_, x0, iter2_, x1, label = data
            if cuda:
                x0 = x0.cuda()
                x1 = x1.cuda()
            output1, output2 = model(x0, x1)
            euclidean_distance = F.pairwise_distance(output1, output2).cpu()
            euclidean_distances.append(euclidean_distance)

    temp = sorted(euclidean_distances)[int(len(euclidean_distances) / 2)]
    for index, data in enumerate(data_loader):
        iter1_, x0, iter2_, x1, label = data
        image_name = iter1_[0] + ' ' + iter2_[0]
        if euclidean_distances[index] < temp:
            result = 0
        else:
            result = 1
        res_fc.append(result)
        res_id.append(image_name)
    return [res_id, res_fc]

def feed_infer(output_file, infer_func):
    prediction_name, prediction_class = infer_func()

    print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_name):
        test_str = name + ' ' + str(prediction_class[index])
        predictions_str.append(test_str)
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=validate_dataloader))

    metric_result = evaluation_metrics(prediction_file, validate_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result

def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def save_model(epochname, model, optimizer, metric_result, train_loss):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict()
    }
    modelname = f'500_{dir_name}_{epochname}_{metric_result}_{train_loss}.pth'
    print('modelname', modelname)
    torch.save(state,  os.path.join(SAVE_PATH, modelname))
    print('model saved: ', modelname)


def load_model(model_name, model, optimizer=None, scheduler=None):
    modelpath = os.path.join(SAVE_PATH, model_name)
    print('modelpath',modelpath)
    state = torch.load(modelpath)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')



if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=2)
    args.add_argument("--lr", type=float, default=0.00006)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--dir_name", type=str, default="29_base1_init2")
    args.add_argument("--model_name", type=str, default="500_29_base1_init2_1_0.9196_0.14946.pth")
    args.add_argument("--prediction_file", type=str, default="prediction_base1_init2_1.txt")
    args.add_argument("--batch", type=int, default=64)
    args.add_argument("--mode", type=str, default="test")
    args.add_argument("--repeat", type=str, default="500")
    

    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode
    repeat = config.repeat
    dir_name = config.dir_name
    

    SAVE_PATH = os.getcwd()
    print('save_path', SAVE_PATH)
    
    # create model

    model = SiameseNetwork()
    if mode == 'test':
        load_model(model_name, model)

    if cuda:
        model = model.cuda()

    if mode == 'train':
        # define loss function
        # loss_fn = nn.CrossEntropyLoss()
        # if cuda:
        #     loss_fn = loss_fn.cuda()

        try:
            os.makedirs(SAVE_PATH)
        except Exception as err:
            print(err)

        class ContrastiveLoss(torch.nn.Module):
            """
            Contrastive loss function.
            Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            """

            def __init__(self, margin=2.0):
                super(ContrastiveLoss, self).__init__()
                self.margin = margin

            def forward(self, output1, output2, label):
                euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
                loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                              (label) * torch.pow(
                    torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

                return loss_contrastive


        # set optimizer

        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
        criterion = ContrastiveLoss()
        # Declare Optimizer

        # get data loader
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch, repeat_num = repeat)
        validate_dataloader, validate_label_file = data_loader(root=DATASET_PATH, phase='validate', batch_size=1)
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)
        #prediction_file = 'prediction.txt'
        counter = []
        loss_history = []
        iteration_number = 0

        # check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter : ", total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params)
        print("------------------------------------------------------------")

        # train
        for epoch in range(num_epochs):
            model.train()
            
            train_losses = []
            for iter_, data in enumerate(train_dataloader, 0):
                iter1_, img0, iter2_, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = model(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                # loss 추가
                train_losses.append(loss_contrastive.item())

                optimizer.step()
                if iter_ % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                        _epoch, num_epochs, loss_contrastive.item(), elapsed, expected))
                    time_ = datetime.datetime.now()

            # scheduler update
            scheduler.step()

            # validate
            metric_result = validate(prediction_file, model, validate_dataloader, validate_label_file, cuda)
            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))
           
             # save model 
            train_loss = round(train_losses[-1], 5)
            metric_result = round(metric_result, 5)
            save_model(str(epoch + 1), model, optimizer, metric_result, train_loss)

          


    elif mode == 'test':
        model.eval()
        # accuracy 확인
        test_dataloader, _ = data_loader(root=DATASET_PATH, phase='test', batch_size=1,repeat_num = 1)
        #prediction_file = "prediction_test.txt"
        prediction_file = f"prediction_{dir_name}.txt"
        test(prediction_file, model, test_dataloader, cuda)