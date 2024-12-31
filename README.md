# sparsify-clip

My aim is to use the COCO dataset to train a Contrastive Language–Image model.

# Pre-requisites

## COCO Dataset

- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)

The COCO dataset will be used to treain the model leveraging the image-caption pairs.

### Download COCO Dataset

The test dataset is not public, we can use the validation-set as test-set

```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip 
```


# Libraries

[pycocotools](https://pypi.org/project/pycocotools/)

