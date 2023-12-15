import models
import datasets
import utils
import trainer
import torch.optim as optim
import numpy as np
import torch
import pandas as pd
import argparse
import copy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from Dynamic_Trajectory_Predictor import video_transforms
import metrics
import torch.nn as nn
from tqdm import tqdm
class LocationDatasetJAAD(Dataset):
    def __init__(self, filename, root_dir, img_root, transform,NUM_FLOW_FRAMES):
        """
        Args:
            filename (string): Pkl file name with data. This must contain the
            optical flow image filename and the label.
            root_dir (string): Path to directory with the pkl file.
        """
        self.df = pd.read_pickle(root_dir + filename)
        print('Loaded data from ',root_dir + filename)
        self.transform = transform
        self.img_root = img_root
        self.NUM_FLOW_FRAMES = NUM_FLOW_FRAMES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Returns:
            sample (dict): Containing:
                flow_stack (np.array):  Stack of optical flow images of shape
                                        256,256,NUM_FLOW_FRAMES*2
                label:                  Label of format [x,x,x...y,y,y...]
                filename
        '''
        # Labels are the CV correction term
        label_x = self.df.loc[idx, 'E_x']
        label_y = self.df.loc[idx, 'E_y']
        NUM_FLOW_FRAMES = self.NUM_FLOW_FRAMES
        filename = self.df.loc[idx, 'Filename']

        label = np.array([label_x,label_y])
        label = label.flatten()

        # Frame number is part of the filename
        frame_num = int(filename.split('_')[2])

        flow_stack = np.zeros((256,256,NUM_FLOW_FRAMES*2)).astype('uint8')

        # Read in the optical flow images
        for frame in range(frame_num+1-NUM_FLOW_FRAMES,frame_num+1):
            frame_name = filename[0:17] + str(frame).zfill(4) + filename[21:]
            img_name_hor = str(self.img_root + 'jaad-horizontal/' + \
                                    frame_name)
            img_name_ver = str(self.img_root + 'jaad-vertical/' + \
                                    frame_name)
            try:
                img_name_hor = '.'.join(img_name_hor.split('.')[0:-1]) + '.jpg'
                img_name_ver = '.'.join(img_name_ver.split('.')[0:-1]) + '.jpg'

                hor_flow = Image.open(img_name_hor).resize((256,256))
                ver_flow = Image.open(img_name_ver).resize((256,256))
            except:
                print('Error: file not loaded. Could not find image file: ')
                print(img_name_hor)
                hor_flow = np.zeros((256,256))
                ver_flow = np.zeros((256,256))

            flow_stack[:,:,int((frame-frame_num-1+NUM_FLOW_FRAMES)*2)] = hor_flow
            flow_stack[:,:,int(((frame-frame_num-1+NUM_FLOW_FRAMES)*2)+1)] = ver_flow

        flow_stack=self.transform(flow_stack)

        sample = {'flow_stack':flow_stack, 'labels': label, 'filenames': filename}
        return sample

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_save_path', help='Path to save the encoder and decoder models')
    parser.add_argument('-data_path', help='Path to bounding box statistics and precomputed DTP features')

    args = parser.parse_args()

    batch_size = 1024
    learning_rate = 1e-3
    weight_decay = 0
    num_workers = 0
    num_epochs = 20
    layers_enc = 1
    layers_dec = 1
    dropout_p = 0
    num_hidden = 512
    normalize = True
    device = torch.device("cuda")
    model_save_path = args.model_save_path
    data_path = args.data_path

    img_root = 'Dynamic_Trajectory_Predictor/data/human-annotated/'
    # Path to training and testing files
    load_path = 'Dynamic_Trajectory_Predictor/data/'
    # CPU or GPU?
    device = torch.device("cuda")

    # Transformers for training and validation
    transform_train = video_transforms.Compose([
            video_transforms.MultiScaleCrop((224, 224), [1.0]),
            video_transforms.ToTensor(),
        ])
    transform_val = video_transforms.Compose([
            video_transforms.Scale((224)),
            video_transforms.ToTensor(),
        ])
    
    
    # # Training settings
    # epochs = 10
    # batch_size = 64
    # learning_rate = 1e-5
    # num_workers = 8
    pretrained = False
    # weight_decay = 1e-2
    NUM_FLOW_FRAMES = 9

    # model_load_path = args.model_load_path
    # model_save_path = args.model_save_path


# testset = LocationDatasetJAAD(filename='jaad_cv_test.pkl',
#                             root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                             shuffle=False, num_workers=num_workers)
# trainset = LocationDatasetJAAD(filename='jaad_cv_train_' + str(fold) + '.pkl',
#                             root_dir=load_path, transform=transform_train, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                             shuffle=True, num_workers=num_workers)
# valset = LocationDatasetJAAD(filename='jaad_cv_val_' + str(fold) + '.pkl',
#                             root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
# val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
#                                             shuffle=False, num_workers=num_workers)

    for detector in ['mask-rcnn']:
        for fold in [1,2,3]:
        
            if pretrained:
                learning_rate = 1e-6
                epochs = 10
            else:
                learning_rate = 1e-5
                epochs = 15
            
            encoder = models.EncoderRNN(device, num_hidden, layers_enc)
            encoder = encoder.to(device)
            encoder = encoder.float()
            decoder = models.DecoderRNN(device, num_hidden, dropout_p, layers_dec)
            decoder = decoder.to(device)
            decoder = decoder.float()
            

            path = data_path    

            print('Training on fold ' + str(fold))
                # print(detector + ' fold ' + str(fold))


            print('Loading data')
            try:    
                    print('1')
                    train_boxes = np.load(path + '/data/jaad_cv_train_1.pkl',allow_pickle=True)
                    val_boxes = np.load(path +  '/data/jaad_cv_val_1.pkl',allow_pickle=True)
                    test_boxes = np.load(path + '/data/jaad_cv_test.pkl',allow_pickle=True)
                    print('2')
                    train_labels = np.load(path  + '/targets_rn18_flow_css_9stack_jaad_fold_1pretrained-False_disp.npy')
                    val_labels = np.load(path + '/targets_rn18_flow_css_9stack_jaad_fold_2pretrained-False_disp.npy')
                    test_labels = np.load(path + '/targets_rn18_flow_css_9stack_jaad_fold_3pretrained-False_disp.npy')
                    print('3')
                    train_dtp_features = np.load(path + '/predictions_rn18_flow_css_9stack_jaad_fold_1pretrained-False_disp.npy')
                    val_dtp_features = np.load(path + '/predictions_rn18_flow_css_9stack_jaad_fold_2pretrained-False_disp.npy')
                    test_dtp_features = np.load(path  + '/predictions_rn18_flow_css_9stack_jaad_fold_3pretrained-False_disp.npy')
   
            except Exception:
                    print('Failed to load data from ' + str(path))
                    exit()

                # # Normalize boxes
                # for i in range(8):
                #     val_boxes[:, i, ] = (val_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
                #         train_boxes[:, i, ].std()
                #     test_boxes[:, i, ] = (test_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
                #         train_boxes[:, i, ].std()
                #     train_boxes[:, i, ] = (train_boxes[:, i, ] - train_boxes[:,
                #                                                             i, ].mean()) / train_boxes[:, i, ].std()

            loss_function = torch.nn.SmoothL1Loss()
            
            testset = LocationDatasetJAAD(filename='jaad_cv_test.pkl',
                                        root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                        shuffle=False, num_workers=num_workers)
            trainset = LocationDatasetJAAD(filename='jaad_cv_train_' + str(fold) + '.pkl',
                                        root_dir=load_path, transform=transform_train, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                        shuffle=True, num_workers=num_workers)
            valset = LocationDatasetJAAD(filename='jaad_cv_val_' + str(fold) + '.pkl',
                                        root_dir=load_path, transform=transform_val, img_root=img_root,NUM_FLOW_FRAMES=NUM_FLOW_FRAMES)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                        shuffle=False, num_workers=num_workers)
            # train_set = datasets.Simple_BB_Dataset(  
            #     train_boxes, train_labels, train_dtp_features)
            # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
            #                                         shuffle=True, num_workers=num_workers)

            # val_set = datasets.Simple_BB_Dataset(
            #     val_boxes, val_labels, val_dtp_features)
            # val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
            #                                         shuffle=False, num_workers=num_workers)
            # test_set = datasets.Simple_BB_Dataset(
            #     test_boxes, test_labels, test_dtp_features)
            # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
            #                                         shuffle=False, num_workers=num_workers)

            optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            def train_seqseq(encoder, decoder, device, train_loader, encoder_optimizer, decoder_optimizer, epoch, loss_function, learning_rate):
                encoder.train()
                decoder.train()
                total_loss = 0
                ades = []
                fdes = []
                for batch_idx, data in enumerate(tqdm(train_loader)):
                    for i in enumerate(tqdm(train_dtp_features)):
                        features, labels, dtp_features = data['flow_stack'], data['labels'].to(device), train_dtp_features.to(device)

                        features = features.float()
                        labels = labels.float()
                        dtp_features = dtp_features.float()

                        context = encoder(features)

                        output = decoder(context, dtp_features, val=False)

                        loss = loss_function(output, labels)

                        ades.append(list(metrics.calc_ade(output.cpu().detach().numpy(), labels.cpu().detach().numpy(), return_mean=False)))
                        fdes.append(list(metrics.calc_fde(output.cpu().detach().numpy(),
                                                        labels.cpu().detach().numpy(), 60, return_mean=False)))

                        # Backward and optimize
                        encoder_optimizer.zero_grad()
                        decoder_optimizer.zero_grad()

                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()

                        # Clip gradients
                        nn.utils.clip_grad_norm(decoder.parameters(), 1)
                        for p in decoder.parameters():
                            p.data.add_(-learning_rate, p.grad.data)
                        total_loss += loss
                    # Flatten lists
                    ades = [item for sublist in ades for item in sublist]
                    fdes = [item for sublist in fdes for item in sublist]

                    print('Train ADE: ', np.round(np.mean(ades), 1))
                    print('Train FDE: ', np.round(np.mean(fdes), 1))
                    print('Train loss: ', total_loss.cpu().detach().numpy())

            def test_seqseq(encoder, decoder, device, test_loader, loss_function, return_predictions=False, phase='Val'):
                encoder.eval()
                decoder.eval()
                ades = []
                fdes = []
                outputs = np.array([])
                targets = np.array([])
                with torch.no_grad():
                    for batch_idx, data in enumerate(tqdm(test_loader)):
                        for i in enumerate(tqdm(train_dtp_features)):
                            features, labels, dtp_features = data['flow_stack'], data['labels'].to(device), test_dtp_features.to(device)
                            features = features.float()
                            labels = labels.float()
                            dtp_features = dtp_features.float()
                            context = encoder(features, val=True)
                            output = decoder(context, dtp_features, val=True)
                            ades.append(list(metrics.calc_ade(output.cpu().numpy(),
                                                            labels.cpu().numpy(), return_mean=False)))
                            fdes.append(list(metrics.calc_fde(output.cpu().numpy(),
                                                            labels.cpu().numpy(), 60, return_mean=False)))
                            if return_predictions:
                                outputs = np.append(outputs, output.cpu().numpy())
                                targets = np.append(targets, labels.cpu().numpy())

                    # Flatten lists
                    ades = [item for sublist in ades for item in sublist]
                    fdes = [item for sublist in fdes for item in sublist]

                    print(phase + ' ADE: ' + str(np.round(np.mean(ades), 1)))
                    print(phase + ' FDE: ' + str(np.round(np.mean(fdes), 1)))

                    return outputs, targets, np.mean(ades), np.mean(fdes)


            best_ade = np.inf
            for epoch in range(num_epochs):
                print('----------- EPOCH ' + str(epoch) + ' -----------')
                print('Training...')
                train_seqseq(encoder, decoder, device, train_loader, optimizer_encoder, optimizer_decoder,
                                    epoch, loss_function, learning_rate)
                print('Validating...')
                val_predictions, val_targets, val_ade, val_fde = trainer.test_seqseq(
                    encoder, decoder, device, val_loader, loss_function, return_predictions=True)
                if epoch == 4:
                    optimizer_encoder = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=weight_decay)
                    optimizer_decoder = optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=weight_decay)
                if epoch == 9:
                    optimizer_encoder = optim.Adam(encoder.parameters(), lr=5e-5, weight_decay=weight_decay)
                    optimizer_decoder = optim.Adam(decoder.parameters(), lr=5e-5, weight_decay=weight_decay)
                if epoch == 14:
                    optimizer_encoder = optim.Adam(encoder.parameters(), lr=2.5e-5, weight_decay=weight_decay)
                    optimizer_decoder = optim.Adam(decoder.parameters(), lr=2.5e-5, weight_decay=weight_decay)
                if val_ade < best_ade:
                    best_encoder, best_decoder = copy.deepcopy(encoder), copy.deepcopy(decoder)
                    best_ade = val_ade
                    best_fde = val_fde
                print('Best validation ADE: ', np.round(best_ade, 1))
                print('Best validation FDE: ', np.round(best_fde, 1))

            print('Saving model weights to ', model_save_path)
            torch.save(encoder.state_dict(), model_save_path + '/encoder_' + detector  + '_gru.weights')
            torch.save(decoder.state_dict(), model_save_path + '/decoder_' + detector + '_gru.weights')

            print('Testing...')
            encoder.eval()
            decoder.eval()
            predictions, targets, ade, fde = test_seqseq(
                encoder, decoder, device, test_loader, loss_function, return_predictions=True, phase='Test')

            print('Getting IOU metrics')

            # Predictions are reletive to constant velocity. To compute AIOU / FIOU, we need the constant velocity predictions.
            test_cv_preds = pd.read_csv('./outputs/constant_velocity/test_' + detector + '.csv')
            results_df = pd.DataFrame()
            results_df['vid'] = test_cv_preds['vid'].copy()
            results_df['filename'] = test_cv_preds['filename'].copy()
            results_df['frame_num'] = test_cv_preds['frame_num'].copy()

            # First 3 columns are file info. Remaining columns are bounding boxes.
            test_cv_preds = test_cv_preds.iloc[:, 3:].values.reshape(len(test_cv_preds), -1, 4)
            predictions = predictions.reshape(-1, 240, order='A')
            predictions = predictions.reshape(-1, 4, 60)

            predictions = utils.xywh_to_x1y1x2y2(predictions)
            predictions = np.swapaxes(predictions, 1, 2)

            predictions = np.around(predictions).astype(int)

            predictions = test_cv_preds - predictions

            gt_df = pd.read_csv('./outputs/ground_truth/test_' + detector + '.csv')
            gt_boxes = gt_df.iloc[:, 3:].values.reshape(len(gt_df), -1, 4)
            aiou = utils.calc_aiou(gt_boxes, predictions)
            fiou = utils.calc_fiou(gt_boxes, predictions)
            print('AIOU: ', round(aiou * 100, 1))
            print('FIOU: ', round(fiou * 100, 1))
