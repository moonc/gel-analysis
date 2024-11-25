# gel-analysis

fetch.ipynb is responsible to extract images from a pdf. You can drop in all pdfs into the "literature" directory and execute the jupyter notebook.


#Pre-Processing 2D Images for TransUnet Analysis
Eventual processing pipeline should look like:
```bash
project/
├── dataset/                          # Root dataset directory
│   ├── raw_images/                   # Original images (e.g., .png, .jpg)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── raw_colored_masks/            # Original colored masks (e.g., .png, .jpg)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── preprocessed_images/          # Preprocessed images as .npz files
│   │   ├── image1.npz
│   │   ├── image2.npz
│   │   └── ...
│   ├── label_masks/                  # Generated label masks as .npz files
│   │   ├── image1.npz
│   │   ├── image2.npz
│   │   └── ...
│   ├── lists/                        # Train/test split files
│   │   ├── train.txt                 # List of sample names for training
│   │   ├── test.txt                  # List of sample names for testing
│   │   └── ...
├── src/                              # Source code for the pipeline
│   ├── preprocess_images.py          # Script to preprocess images
│   ├── generate_masks.py             # Script to generate label masks
│   ├── dataset_class.py              # Dataset class for PyTorch
│   ├── verify_masks.py               # Script to verify the pipeline
│   └── ...
├── outputs/                          # Outputs from training/inference
│   ├── logs/                         # Logs for training
│   ├── checkpoints/                  # Model checkpoints
│   ├── predictions/                  # Predicted masks for test images
│   │   ├── image1_prediction.png
│   │   ├── image2_prediction.png
│   │   └── ...
│   └── ...
└── README.md                         # Documentation for the pipeline
```
Please note the majority of this infrastructure is derived from the original TransUnet repository.
