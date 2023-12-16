We have referred through the following links in order to develop this model:

1. olly-styles/Multiple-Object-Forecasting: Repo for the paper 'Multiple Object Forecasting: Predicting Future Object Locations in Diverse Environments'. WACV 2020
Olly-Styles
https://github.com/olly-styles/Multiple-Object-Forecasting/tree/master

2. olly-styles/Dynamic-Trajectory-Predictor: Repo for the paper 'Forecasting Pedestrian Trajectory with Machine-Annotated Training Data'. IV 2019
Olly-Styles
https://github.com/olly-styles/Dynamic-Trajectory-Predictor/tree/master


# Pedestrain_Movement_Prediction
Please clone this repository https://github.com/Tharun2331/Dynamic_Trejectory_Predictor/tree/master first then follow below steps.
This part of the Pedestrian Movement Predection uses the dynamic trajectory prediction using the JAAD Dataset.

#Instructions

1. Install the required packages using: pip install -r requirements.txt

2. Download the JAAD folder through the command:
    gdown https://drive.google.com/uc?id=1OuXLKrB6ItikYbnCM1yODQk2IUAmQ07y
    gdown https://drive.google.com/uc?id=1mP4y-S8NEnavfGGZLCzkfw4EIpDUfJnp
   Then unzip the human annotated zip file using: unzip human-annotated.zip

3. To preprocess the dataset, execute the following programs:
   python process_bounding_boxes_jaad.py
   python compute_cv_correction_jaad.py

4. To train the model, execute:
   python train_dtp_jaad.py

5. To fine tune the pre-trained model, execute the following:
   cd data && mkdir models && cd models
   gdown https://drive.google.com/uc?id=1J2VclWeEjMj7WQhTmEPhjCaza4w5PSmX
   python train_dtp_jaad.py -1 ./data/models/bdd10k_rn18_flow_css_9stack_training_proportion_100_shuffled_disp.weights

6. To run the BBD, you can either use Yolov3 or Faster-RCNN

   For Yolov3:
   cd data
   gdown https://drive.google.com/uc?id=17Fvkrtxg_NEH2edH-wEp_Po5Y777zGQJ
   gdown https://drive.google.com/uc?id=1mcL-c-FT19ePFdaLu8v1rmApoLDIeGYe
   unzip yolov3.zip
   python process_bounding_boxes_bdd.py
   python compute_cv_correction_bdd.py  
   python train_dtp_bdd.py

   For Faster-RCNN:
   cd data
   gdown https://drive.google.com/uc?id=1SNVe9SSRYiG-6WQZpvIOl_KtxfAThG2y
   gdown https://drive.google.com/uc?id=1hKbnGThFS-shggFraQMuGhl9gy0E7ylV
   unzip faster-rcnn.zip
   cd ../preprocessing
   python process_bounding_boxes_bdd.py -d faster-rcnn
   python compute_cv_correction_bdd.py -d faster-rcnn
   python train_dtp_bdd.py -d faster-rcnn



