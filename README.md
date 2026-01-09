# mlops_group54
Project repository for MLOps 2026 for group 54.

# Project Description
The goal of this project is to build a machine learning system that can classify brain MRI images into different tumor categories, including a class for “no tumor”.
The project is not only about training a model with high accuracy. The main focus is on building a reliable and reproducible workflow, where every step of the process is clearly defined and can be repeated in the same way. This means that the entire pipeline from raw data to a trained model should be easy to understand, run, and reproduce.

Data set:
The chosen data set is a medical data set on brain tumor classification between 3 different brain tumors or no brain tumor, meaning a total of 4 classes. The data has been gathered by Sartaj Bhuvaji and has been downloaded from Kaggle. The dataset consists of 3262 MRI images, each labelled with the true class, making this a supervised learning problem. The data set is already split into training and testing, but the classes are not balanced, meaning the classes are not of equal sizes. As the data set consists of images, the set has to be preprocessed into a numerical representation. We also have to consider, if the classes need to be balanced, depending on how remarkable the imbalance is. Moreover, as the data is currently divided into the labelled folders, the data also has to be shuffled before using it in training and testing. 

Models:
For solving the classification task, we expect to implement a convolutional neural network. MRI images and classifying tumors is a complex problem, where others have proven great success working with complex models detecting non-linear patterns. We have chosen to use a pretrained CNN from TorchVision from PyTorch as the model. Specifically, we considered resnet-15 which is a good backbone for image classification. This model will then be trained, on our specific data set, where the weights will be fine tuned to be able to make a classification on our classification problem also using torch. Evaluating the performance of the model will be done using the test set 


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
