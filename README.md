Larger resolution face masked, weirdly warped, deepfake, requires https://github.com/keras-team/keras-contrib.git to be cloned into the root of the repo and A and B's alignments.json to be copied into the correct /data/* folder before training.

* Clone this repository
* cd into the folder df and clone https://github.com/keras-team/keras-contrib.git
* make sure that the folder keras_contrib is in the root of the df respository
* run align_images_masked on your source A and B image folders.
* copy the aligned cropped images into the A or B folder along with the alignments.json files from the source image folders.
* run train.py as usual
* wait
* run merge_faces_larger.py on your image folder.
