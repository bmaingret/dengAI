# DengAI: Predicting Disease Spread

This repo contains the work realized during my participation to a DrivenData Challenge: [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread).

This repository also serve as a submission repository for the [Udacity Machine Learning Engineer](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) - Capstone Project

The  project was done on [AWS SageMaker](https://aws.amazon.com/sagemaker). Although some notebooks can be run on any machine with Jupyter installed, the training and hypertuning of the model can only be done on AWS platform.

## Files

    .
    ├── data/*.csv            # Scores and hyperparameters of trained models and submissions
    ├── data/figures/         # Comparison on validation data set
    ├── data/input/           # Copy of the original data set
    ├── data/submissions/     # Some of the submitted files to the competition
    ├── report/               # Udacity Machine Learning Engineer Capstone proposal and final report
    ├── src/preprocessing.py  # Preprocessing script used on SageMaker
    └── *.ipynb               # Notebooks used for EDA , model training, validation and submission creation , hypertuning job creation, hypertuning analysis, comparison of imputation method, and scores and submissions analysis

## Setup

### Setup you SageMaker environment (nothing fancy here)

### Custom docker image for SageMaker

To be able to use a newer version of scikit-learn (0.22.2) than the one provided on SageMaker (0.20.0), we create a docker container with updated version and uploaded it to an AWS ECR repository.

All the information to create the container as well as ready-to-use Dockerfile with scikit-learn v0.22.0 can be found on this [repository](https://github.com/bmaingret/sagemaker-scikit-learn-container) (fork of [sagemaker-scikit-learn-container](https://github.com/aws/sagemaker-scikit-learn-container)).


Create a [repository in Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html).

Once created you can access it from the AWS console and select `View push commands` to easily be able to push your created container to the repository.

Once uploaded you can reference it in `1-deepAr.ipynb` for the `uri` parameter while creating the `ScriptProcessor`:

```python
sklearn_processor = ScriptProcessor(image_uri='AWS-ID.dkr.ecr.eu-central-1.amazonaws.com/ECS-REPOSITORY:IMAGE-TAG',
                                     role=role,
                                     instance_type='ml.m4.xlarge',
                                     instance_count=1,
                                     command = ["python3"], # default required using the same as in SKLearnProcessor
                                     volume_size_in_gb=30, # default required using the same as in SKLearnProcessor
                                     base_job_name=f'{prefix}-{tag}-pprocess'
                                    )
```

## PDF generation

PDF files were generated from Markdown files using pandoc:

 `pandoc -o output.pdf input.md -V geometry:margin=1in`
