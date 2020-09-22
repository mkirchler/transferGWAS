# Export image embeddings at specified layer in convolutional neural network

This directory contains code to condense images into low-dimensional embeddings.

Run via
```bash
python export_embeddings.py PATH_TO_CSV OUT_DIR \
                --img_size 448 \
                --tfms tta \
                --dev cuda:0 \
                --n_pcs 50 \
                --model resnet50 \
                --pretraining imagenet \ # or e.g. "../models/eyepacs_pretrained.pt"
                --layer L2 L3 L4
```
where `PATH_TO_CSV` is a `.csv` files with `IID` and `path` column (see `Input format` below), and `OUT_DIR` is the directory to save the results.

### TODO Python requirements


Install the conda environment (same as in the pretraining directory) via
```bash
conda ...
```

### Input format

If you have a single image per individual, simply create a `.csv` file with an `IID` and `path` column as in this example:
```CSV
IID,path
1234567,path/to/img1.png
2345678,path/to/img2.png
3456789,path/to/img3.png
4567890,path/to/img4.png
...
```

If each individual has more than one image (such as one image per eye, multiple slices from a scan, ...), create an additional `instance` column. Note that all individuals need to have the same number of instances (otherwise they will be sorted out).
```CSV
IID,path,instance
1234567,path/to/left_img1.png,left
1234567,path/to/right_img2.png,right
2345678,path/to/left_img3.png,left
2345678,path/to/right_img4.png,right
```

Images must be in a standard image format that `PIL` can read (such as `.png` or `.jpg`). If your data is in another format and you don't want to convert it, you need to extend the `_load_item` method in the `ImageData` class to read in your data format.