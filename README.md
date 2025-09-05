# MED3pa: Predictive Performance Precision Analysis in Medicine

## Table of Contents
- [Overview](#overview)
- [Key Functionalities](#key-functionalities)
- [Subpackages](#subpackages)
- [Getting Started with the Package](#getting-started)
    - [Installation](#installation)
    - [A Simple Example](#a-simple-example)
- [Acknowledgement](#acknowledgement)
- [References](#references)
- [Authors](#authors)
- [Statement](#statement)
- [Supported Python Versions](#supported-python-versions)

## Overview

<img src="https://github.com/MEDomics-UdeS/MED3pa/blob/review/docs/diagrams/package_white_bg.svg" alt="Overview" style="width:100%;">

The **MED3pa** package is specifically designed to address critical challenges in deploying machine learning models, particularly focusing on the robustness and reliability of models under real-world conditions. It provides comprehensive tools for evaluating model stability and performance in the face of **covariate shifts**, **uncertainty**, and **problematic data profiles**.

## Key Functionalities

- ** Model Confidence Estimation**: Through the MED3pa subpackage, the package measures the predictive confidence at both individual and group (profile) levels. This helps in understanding the reliability of model predictions and in making informed decisions based on model outputs.

- **Identification of Problematic Profiles**: MED3pa analyzes data profiles for whom the BaseModel consistently leads to poor model performance. This capability allows developers to refine training datasets or retrain models to handle these edge cases effectively.

## Subpackages

<p align="center">
    <img src="https://github.com/MEDomics-UdeS/MED3pa/blob/review/docs/diagrams/subpackages.svg" alt="Overview">
</p>

The package is structured into four distinct subpackages:

- **datasets**: Stores and manages the dataset.
- **models**: Handles ML models operations.
- **med3pa**: Evaluates the model’s performance & extracts problematic profiles.

This modularity allows users to easily integrate and utilize specific functionalities tailored to their needs without dealing with unnecessary complexities.

## Getting Started with the Package

To get started with MED3pa, follow the installation instructions and usage examples provided in the documentation.

### Installation

```bash
pip install MED3pa
```

### A simple exemple
We have created a [simple example](https://github.com/MEDomics-UdeS/MED3pa/tree/main/examples) of using the MED3pa package. 
[See the full example here](https://github.com/MEDomics-UdeS/MED3pa/tree/main/examples/oym_example.ipynb)
```python
from MED3pa.datasets import DatasetsManager
from MED3pa.med3pa import Med3paExperiment
from MED3pa.models import BaseModelManager
from MED3pa.visualization.mdr_visualization import visualize_mdr
from MED3pa.visualization.profiles_visualization import visualize_tree

...

# Initialize the DatasetsManager
datasets = DatasetsManager()
datasets.set_from_data(dataset_type="testing",
                       observations=x_evaluation.to_numpy(),
                       true_labels=y_evaluation,
                       column_labels=x_evaluation.columns)
# Initialize the BaseModelManager
base_model_manager = BaseModelManager(model=clf)

# Execute the MED3PA experiment
results = Med3paExperiment.run(
    datasets_manager=datasets,
    base_model_manager=base_model_manager,
    **med3pa_params
)

# Save the results to a specified directory
results.save(file_path='results/oym')

# Visualize results
visualize_mdr(result=results, filename='results/oym/mdr')
visualize_tree(result=results, filename='results/oym/profiles')

```

## Acknowledgement
MED3pa is an open-source package developed at the [MEDomicsLab](https://www.medomicslab.com/) laboratory. We welcome any contribution and feedback. 

## Authors
* [Olivier Lefebvre: ](https://www.linkedin.com/in/olivier-lefebvre-bb8837162/) Student (Ph. D. Computer science) at Université de Sherbrooke
* [Lyna Chikouche: ](https://www.linkedin.com/in/lynahiba-chikouche-62a5181bb/) Research intern at MEDomics-Udes laboratory.
* [Ludmila Amriou: ](https://www.linkedin.com/in/ludmila-amriou-875b58238//) Research intern at MEDomics-Udes laboratory.
* [Martin Vallières: ](https://www.linkedin.com/in/martvallieres/) Associate professor, Department of Oncology at McGill University

## Statement

This package is part of https://www.medomics.ai/, a package providing research utility tools for developing precision medicine applications.

```
Copyright (C) 2024 MEDomics consortium

GPLV3 LICENSE SYNOPSIS

Here's what the license entails:

1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
```

More information about the [LICENSE can be found here](https://github.com/MEDomics-UdeS/MEDimage/blob/main/LICENSE.md)

## Supported Python Versions

The **MED3pa** package is developed and tested with Python 3.12.3.

Additionally, it is compatible with the following Python versions:
- Python 3.11.x
- Python 3.10.x
- Python 3.9.x

While the package may work with other versions of Python, these are the versions we officially support and recommend.
