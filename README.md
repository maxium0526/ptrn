# Projective Transformation Rectification Network Official Implementation

<https://arxiv.org/abs/2210.05954>

### Abstract

Classification on smartphone-captured chest X-ray (CXR) photos to detect pathologies is challenging due to the projective transformation caused by the non-ideal camera position. Recently, various rectification methods have been proposed for different photo rectification tasks such as document photos, license plate photos, etc. Unfortunately, we found that none of them is suitable for CXR photos, due to their specific transformation type, image appearance, annotation type, etc. In this paper, we propose an innovative deep learning-based Projective Transformation Rectification Network (PTRN) to automatically rectify CXR photos by predicting the projective transformation matrix. To the best of our knowledge, it is the first work to predict the projective transformation matrix as the learning goal for photo rectification. Additionally, to avoid the expensive collection of natural data, synthetic CXR photos are generated under the consideration of natural perturbations, extra screens, etc. We evaluate the proposed approach in the CheXphoto smartphone-captured CXR photos classification competition hosted by the Stanford University Machine Learning Group, our approach won first place with a huge performance improvement (ours 0.850, second-best 0.762, in AUC). A deeper study demonstrates that the use of PTRN successfully achieves the classification performance on the spatially transformed CXR photos to the same level as on the high-quality digital CXR images, indicating PTRN can eliminate all negative impacts of projective transformation on the CXR photos.

![ptrn](https://user-images.githubusercontent.com/38188772/229279128-0da2e79c-4b10-4aff-81a6-19540e649682.png)

### Requirements

- Python 3 (3.8.5)
- TensorFlow 2 (2.6.0)
- Pillow (8.4.0)
- OpenCV Python (4.5.3.56)
- imgaug (0.4.0)
- Pandas (1.3.1)
- Shapely (1.7.1)
- SKlearn (0.0)

### Strat with Virtual Environment

Install the requirements with requirements.txt

`pip install -r requirements.txt`
  
### Train your own PTRN model

1. ~~Use labelme.exe to mark the 4 vertices of each validation photographs in CheXphoto.~~  Annotations are available now, just merge it to the CheXphoto dataset.
2. Open train.py, fill in the path of your datasets (CheXphoto, MS COCO, etc.)
3. `python train.py`

### Evaluate the trained PTRN on CheXphoto-Monitor and CheXphoto-Film validation set

In the paper, we split the CheXphoto validation set into two subsets (CheXphoto-Monitor and CheXphoto-Film).

1. Go the your dataset folder that contains your CheXphoto dataset.
2. Create two new folders called `CheXphoto-natural` and `CheXphoto-film` respectively.
3. Copy `CheXphoto-v1.1/valid.csv` to `CheXphoto-natural/valid.csv` and `CheXphoto-film/valid.csv` respectively.
4. Open `CheXphoto-natural/valid.csv`, delete the unrelated rows. Only the rows of the 234 natural photographs are remained.
5. Open `CheXphoto-film/valid.csv`, delete the unrelated rows. Only the rows of the 250 film photographs are remained.
6. Open evaluate.py, fill in the path of your datasets.
7. `python evaluate.py`

### Use PTRN to rectify a CXR photograph

1. `python use.py your_img.jpg`

### Training Demo

|                	| CheXphoto-Monitor (validation) 	| CheXphoto-Film (test) 	|
|----------------	|--------------------------------	|-----------------------	|
| Paper Reported 	| 0.942                          	| 0.892                 	|
| This code      	| **0.945**                     	| **0.932**             	|


![2021-11-22 170029](https://user-images.githubusercontent.com/38188772/142833946-99d1a506-5326-4f1b-be9e-66ea0cdd202a.png)

![2021-11-22 170429](https://user-images.githubusercontent.com/38188772/142833963-ff88cb89-5c3f-4350-b12c-db6d28a2dcab.png)

We uploaded this trained PTRN instance on Google Cloud Drive. It is now available for download. (Note. this instance is ***NOT*** used or reported in the paper)

<https://drive.google.com/drive/folders/1_ypUw9-XHdnzxxh0QP81IxAzFZIDr35-?usp=share_link>
