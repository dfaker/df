## df

Larger resolution face masked, weirdly warped, deepfake, requires https://github.com/keras-team/keras-contrib.git to be cloned into the root of the repo and A and B's alignments.json to be copied into the correct /data/* folder before training.

Inputs are 64x64 images outputs are a pair of 128x128 images one RGB with the reconstructed face, one B/W to be used as a mask to guide what sections of the image are to be replaced. 

For the reconstrcuted face masked DSSIM loss is used that behaves as a standard SSIM difference measure in the central face area and always returns zero loss in the surrounding background area outside of the face so as as not to train irrelevant features.

MSE is used for the mask.

### Training
![training](https://github.com/dfaker/df/raw/master/trumpcage.png)

### Merged
![merged](https://github.com/dfaker/df/raw/master/trumpcagemerge.jpg)

### Guide

* Clone this repository
* cd into the folder df and clone https://github.com/keras-team/keras-contrib.git
* make sure that the folder keras_contrib is in the root of the df respository
* run align_images_masked on your source A and B image folders.
* copy the aligned cropped images into the A or B folder along with the alignments.json files from the source image folders.
* run train.py as usual
* wait
* run merge_faces_larger.py on your image folder.

Directory structure for training data should look like (image names for example purposes only):

    
    df
    │
    └───data
        │
        ├───A
        │   │ alignments.json
        │   │ trainingimageA1.jpg
        │   │ trainingimageA2.jpg
        │
        ├───B
        │   │ alignments.json
        │   │ trainingimageB1.jpg
        │   │ trainingimageB2.jpg
        


or as a file listing:



    .\df\data\A\alignments.json
    .\df\data\A\trainingimageA1.jpg
    .\df\data\A\trainingimageA2.jpg
    .\df\data\B\alignments.json
    .\df\data\B\trainingimageB1.jpg
    .\df\data\B\trainingimageB2.jpg
