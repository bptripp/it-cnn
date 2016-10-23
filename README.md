# it-cnn
This project has two purposes: 1) to compare activity in the second-last layers of deep object-recognition convolutional neural networks (CNNs) with activity in the primate inferior temporal cortex (IT), and 2) to train such CNNs to emulate IT activity. 

Certain parts of the code expect images that aren't included in the repository. 

First, tuning/selectivity.py needs images extracted from the the supplementary material PDF of Lehky et al. (2011), Statistics of visual responses in primate inferotemporal cortex to object stimuli. I think it would stretch fair use excessively to reproduce these. Second, training/orientation.py needs many training images that are generated in Unity. I haven't posted these anywhere, but I could on request. Third, the scripts in tuning/ generally need images that are produced by training/cnn_stimuli.py. This code produces many stimuli from source images found in tuning/source-images, so run it first. 

