# Projective Transformation Rectification Network

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

1. Put the ground truth CXR region labels (in `labels/`) in the CheXphoto dataset.
2. Open train.py, fill in the path of your datasets (CheXphoto, MS COCO, etc.)
3. `python train.py`

### Evaluate your trained PTRN on CheXphoto-Monitor and CheXphoto-Film validation set

In the paper, we split the CheXphoto validation set into two subsets (CheXphoto-Monitor and CheXphoto-Film).

1. Go the your dataset folder that contains your CheXphoto dataset.
2. Create two new folders called `CheXphoto-natural` and `CheXphoto-film` respectively.
3. Copy `CheXphoto-v1.1/valid.csv` to `CheXphoto-natural/valid.csv` and `CheXphoto-film/valid.csv` respectively.
4. Open `CheXphoto-natural/valid.csv`, delete the unrelated rows. Only the 234 natural photographs is remained.
5. Open `CheXphoto-film/valid.csv`, delete the unrelated rows. Only the 500 film photographs is remained.
6. Open evaluate.py, fill in the path of your datasets.
7. `python evaluate.py`

### Use PTRN to rectify a CXR photograph

1. `python use.py your_img_dir.jpg`

### Experiment Result

|                	| CheXphoto-Monitor (validation) 	| CheXphoto-Film (test) 	|
|----------------	|--------------------------------	|-----------------------	|
| Paper Reported 	| 0.942                          	| 0.892                 	|
| This code      	| **0.945**                     	| **0.932**             	|


![螢幕擷取畫面 2021-11-22 170029](https://user-images.githubusercontent.com/38188772/142833946-99d1a506-5326-4f1b-be9e-66ea0cdd202a.png)

![螢幕擷取畫面 2021-11-22 170429](https://user-images.githubusercontent.com/38188772/142833963-ff88cb89-5c3f-4350-b12c-db6d28a2dcab.png)
