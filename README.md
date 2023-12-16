# References

1. olly-styles/Multiple-Object-Forecasting: Repo for the paper 'Multiple Object Forecasting: Predicting Future Object Locations in Diverse Environments'. WACV 2020 Olly-Styles https://github.com/olly-styles/Multiple-Object-Forecasting/tree/master

2. olly-styles/Dynamic-Trajectory-Predictor: Repo for the paper 'Forecasting Pedestrian Trajectory with Machine-Annotated Training Data'. IV 2019 Olly-Styles https://github.com/olly-styles/Dynamic-Trajectory-Predictor/tree/master


# Pedestrain_Movement_Prediction
This project is to demonstrate the prediction of the pedestrian movement. This is extremely useful for various applications such as autonomous driving cars and also for people with impaired vision to anticipate any objects or people coming into their paths.

The model consists of encoders which uses the Gated Recurring Unit (GRU) and another GRU for decoders to produce the predicted bounding box. It also uses the ResNet50 for the optical flow feature encoder.

# Instructions to implement:

1. To use the Citywalks dataset, use:
   gdown https://drive.google.com/uc?id==1oMN-fsWvEjUZ9Ah_3JwUuIY7cmR0OP_Q

2. We used Mask-RCNN to detect the pedestrians and DeepSort to track the pedestrians. The tracking results can be downloaded using:
   gdown https://drive.google.com/uc?id=12-_FiphR5m0Yd455pem13OVnvCvi-yIn

3. We can use the weather,time and day metadata. Use:
   gdown https://drive.google.com/uc?id=163VmkvnB_ruTDUTezsz1jxBJtdoH98Vc

4. For the cross validation split, for each fold we will use a training set, validation set and testing set, and will be different for every fold performed
   
5. Use the average displacement error (ADE), final displacement error (FDE), the average intersection over union (AIOU) and the final intersection over union (FIOU) metrics in order to evaluate the movemebnt prediction model on Citywalsk dataset. To use the ground truth trajectories and predictions from STED, use:
  gdown https://drive.google.com/uc?id=1PXzIOHGScWqhAWXnO4q70fch8wF1DJUx

  Then to evaluate the predictions, execute the evaluate_outputs.py program.
  python evaluate_outputs.py -gt ./outputs/ground_truth/ -pred ./outputs/sted/

6. Install the required libraries by executing the command:
   pip install -r requirements.txt

7. Download the pre-trained models of each fold of Citywalks, using:
   gdown https://drive.google.com/uc?id=1GGIZEJlD5wNNgCqafevraMJS71AyJ3nF

8. Download the DTP features for Mask-RCNN using:
   gdown https://drive.google.com/uc?id=1LBTJrmU7FSQIxM8mcVl4Vl9MN0_x7ccQ

9. Now we need to train the model and compute the AIOU and FIOU metrics by using:
    python sted_train.py -model_save_path PATH_TO_SAVE_MODEL -data_path PATH_TO_FEATURES

10. Now evaluate the model by using:
    python sted_train.py -model_save_path PATH_TO_SAVE_MODEL -data_path PATH_TO_FEATURES

11. Once the model has been trained and evaluate, you can execute the results.py program to get the results to produce the predicted bounding box on the video.






