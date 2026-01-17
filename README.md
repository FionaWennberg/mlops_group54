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

# Status:
- I have added wandb, which will be used for logging, when we have a training code 


# Check-liste status:

Week 1:
Completed:
•	Create a git repository (M5)
•	Make sure that all team members have write access to the GitHub repository (M5)
•	Create a dedicated environment for you project to keep track of your packages (M2)
•	Create the initial file structure using cookiecutter with an appropriate template (M6)
•	Fill out the data.py file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
•	Add a model to model.py and a training procedure to train.py and get that running (M6)
•	Remember to fill out the requirements.txt and requirements_dev.txt file with whatever dependencies that you are using (M2+M6)
•	Remember to comply with good coding practices (pep8) while doing the project (M7)
•	Setup version control for your data or part of your data (M8)
•	Construct one or multiple docker files for your code (M10)
•	Build the docker files locally and make sure they work as intended (M10)
•	Write one or multiple configurations files for your experiments (M11)
•	Used Hydra to load the configurations and manage your hyperparameters (M11)
•	Use logging to log important events in your code (M14)
•	Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)

Not completed:
•	Do a bit of code typing and remember to document essential parts of your code (M7)
•	Add command line interfaces and project commands to your code where it makes sense (M9)
•	Use profiling to optimize your code (M12)
•	Consider running a hyperparameter optimization sweep (M14)
•	Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)


Week 2:
Completed:
•	Write unit tests related to the data part of your code (M16)
•	Write unit tests related to model construction and or model training (M16)
•	Calculate the code coverage (M16)
•	Get some continuous integration running on the GitHub repository (M17)
•	Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
•	Add a linting step to your continuous integration (M17)
•	Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
•	Create a FastAPI application that can do inference using your model (M22)
•	Write API tests for your application and setup continues integration for these (M24)

Not completed:
•	Add pre-commit hooks to your version control setup (M18)
•	Add a continues workflow that triggers when data changes (M19)
•	Add a continues workflow that triggers when changes to the model registry is made (M19)
•	Create a trigger workflow for automatically building your docker images (M21)
•	Get your model training in GCP using either the Engine or Vertex AI (M21)
•	Deploy your model in GCP using either Functions or Run as the backend (M23)
•	Load test your application (M24)
•	Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
•	Create a frontend for your API (M26)

Week 3:
Completed:

Not completed:
•	Check how robust your model is towards data drifting (M27)
•	Setup collection of input-output data from your deployed application (M27)
•	Deploy to the cloud a drift detection API (M27)
•	Instrument your API with a couple of system metrics (M28)
•	Setup cloud monitoring of your instrumented application (M28)
•	Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
•	If applicable, optimize the performance of your data loading using distributed data loading (M29)
•	If applicable, optimize the performance of your training pipeline by using distributed training (M30)
•	Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

