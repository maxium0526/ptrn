# Projective Transformation Rectification Network Official Implementation

### Paper

[Published Paper](https://doi.org/10.1016/j.compbiomed.2023.107277)

[Preprint](https://arxiv.org/abs/2210.05954)

#### Image projective transformation rectification with synthetic data for smartphone-captured chest X-ray photos classification

Automatic interpretation of chest X-ray (CXR) photos taken by smartphones at the same performance level as with digital CXRs is challenging, due to the projective transformation caused by the non-ideal camera position. Existing rectification methods for other camera-captured photos (document photos, license plate photos, etc.) cannot precisely rectify the projective transformation of CXR photos, due to its specific projective transformation type. In this paper, we propose an innovative deep learning-based Projective Transformation Rectification Network (PTRN) to automatically rectify the projective transformation of CXR photos by predicting the projective transformation matrix. Additionally, synthetic CXR photos are generated for training with the consideration of visual artifacts of natural images. The effectiveness of the proposed classification pipeline with PTRN is evaluated in the CheXphoto smartphone-captured CXR photo classification competition. It achieves first place with a huge performance improvement (ours 0.850, second-best 0.762, in AUC). Moreover, experimental results show that our approach successfully achieves the same performance level of digital CXR classification (AUC 0.893) on CXR photo classification (AUC 0.893).

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
