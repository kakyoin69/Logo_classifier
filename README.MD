# Logo recognition project

## Overview
This project is designed to recognition logos in images using EfficientNet and Dataset Logo-2K+

## Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Configuration](#configuration)
- [Training EfficientNet](#trainingefficientnet)
- [How It Works](#howitworks)
- [Example Output](#exampleoutput)
- [Contacts](#contacts)


## Features

- Classification of similar brand logos

- Outputs matching brand names and similar images

## Installation
Clone:
```bash
https://github.com/kakyoin69/Logo_classifier.git
```
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```
Additionally, install EfficientNet and OpenAI CLIP:
```bash
pip install efficientnet_pytorch
pip install git+https://github.com/openai/CLIP.git
```
## Usage

* To start a search for logos of the same brand, use the following command:
```bash
python main.py --image "path/to/image.jpg" --images 'path/to/image.jpg, path/to/image1.jpg...etc' --weights "path/to/efficientnet_logo2k.pth" --classes "path/to/classes.txt"
```

* Using a single image with text:
```bash
python main.py --text "Is it Mazda?" --image "path/to/image.jpg" --weights "path/to/efficientnet_logo2k.pth" --classes "path/to/classes.txt"
```
## Arguments:

- ```--image```: Path or URL to the image file

- ```--images```: Path to a folder or URL images 

- ```--text```: Optional text question (e.g., "Is it Mazda?")

- ```--classes```: Path to the classes.txt file containing brand names

- ```--weights```: Path to the trained EfficientNet model (efficientnet_logo2k.pth)
## Configuration
You can also use the configuration file ***config/config.yaml.***:
```markdown
config:
  weights: "path/to/efficientnet_logo2k.pth"
  classes: "path/to/classes.txt"
  image: "path/to/image.jpg"
  images: "path/to/images_folder/"
  text: "Question" #example "Is it Mazda?"
```

## Training EfficientNet
The EfficientNet model, which was trained on the Logo-2k dataset, was used as a classifier. The working block is located in the ***ml_experements/logo_classifier.ipynb***

## How It Works

1. The script loaded the configuration parameters and arguments.
2. The data loader processes the input images and prepares them for analysis.
3. The process manager uses Efficient Net to classify logos.
4. The CLIP is used to verify the brand.
5. The detected logos are compared with the original one.
6. The system will return paths or links of matching logos.

## Example Output
```markdown
2025-02-04 20:38:31.970 | INFO     | __main__:process_answer:67 - Logo: Mazda || Similar images:
2025-02-04 20:38:31.970 | INFO     | __main__:process_answer:70 -  image: https://m.media-amazon.com/images/I/718vwiv9QlL._AC_UF1000,1000_QL80_.jpg
2025-02-04 20:38:31.970 | INFO     | __main__:process_answer:78 - No, this is a different logo.
```

## Contact
Made specifically for the VK ML task
tg @flopperator

