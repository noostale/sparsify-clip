# sparsify-clip

My aim is to use the COCO dataset to train a Contrastive Language–Image model.

## Conda Environment

The conda environment can be created using the following command:

```bash
conda env create -f environment.yml
```

This will create a new environment called `sparsify-clip`.


## COCO Dataset

- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)

The COCO dataset will be used to treain the model leveraging the image-caption pairs.

## Download COCO Dataset

The test dataset is not public, we can use the validation-set as test-set

The dataset can be downloaded using the following commands:

```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

The data folder should look like this:

```bash
├── data
│   └── coco
│       ├── annotations (extract here the annotations_trainval2017.zip)
│       └── images (extract here the train2017.zip and val2017.zip)
```


## Metrics

- `cosine similarity`: measures how similar two vectors are in terms of their orientation in a multi-dimensional space, regardless of their magnitude.

- `angular value`: inverse of the cosine similarity, it measures the angle between two vectors.

