![Workshop Banner](https://github.com/CLDiego/uom_fse_dl_workshop/blob/main/figs/banner.png)

[![Version](https://img.shields.io/badge/version-1.5.0-blue?logo=github)](https://github.com/CLDiego/uom_fse_dl_workshop)
<!-- [![GitHub Pages](https://img.shields.io/badge/View%20Site-GitHub%20Pages-blue?logo=github)](https://cldiego.github.io/uom_fse_dl_workshop/) -->

# Deep Learning with PyTorch – Workshop

## Overview 📋

This hands-on workshop introduces the fundamentals of deep learning using PyTorch. Participants will learn by building real models and solving practical tasks across two sessions at **Nancy Rothwell – 2A.011 M&T**, on **23 February** and **2 March 2025**.

### What You'll Learn

* Core PyTorch concepts (tensors, autograd, GPU usage)
* Building and training artificial neural networks and autoencoders
* Implementing CNNs for image classification tasks
* Applying transfer learning with pre-trained models for image segmentation
* Working with real-world datasets for classification, regression, and anomaly detection
* Understanding data preprocessing, augmentation, and normalisation techniques

---

## Getting Started 🛠️

### ✅ Recommended Platform: [Google Colab](https://colab.research.google.com/)

Colab provides a free, GPU-enabled environment and is the **primary platform** for this workshop. No local installation is required.

#### What You Need

* A Google account
* Reliable internet connection

#### Running the Notebooks on Colab

1. Open the GitHub repo and click the **"Open in Colab"** badge at the top of any notebook, or:
   * Download the notebook locally.
   * Open [Google Colab](https://colab.research.google.com/).
   * Use **File > Upload Notebook** to load it.
2. Enable GPU: **Runtime > Change runtime type > Hardware Accelerator > GPU**
3. Run the first setup cell to install all required dependencies.

📘 [Colab Tips](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) | [Colab FAQ](https://research.google.com/colaboratory/faq.html)

---

### 💻 Running Locally (Optional)

> **Note:** Local setup is optional. Google Colab is strongly preferred for the workshop.

#### Requirements

* Python 3.14+
* `pip` or [`uv`](https://docs.astral.sh/uv/) (recommended for faster installs)

#### Setup Steps

**Option A – using `uv` (recommended):**

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Clone the repository
git clone https://github.com/CLDiego/uom_fse_dl_workshop.git
cd uom_fse_dl_workshop

# 3. Create a virtual environment and install all dependencies
uv sync

# 4. Launch Jupyter
uv run jupyter notebook
```

**Option B – using `pip`:**

```bash
# 1. Clone the repository
git clone https://github.com/CLDiego/uom_fse_dl_workshop.git
cd uom_fse_dl_workshop

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install the project and all dependencies (from pyproject.toml)
pip install .

# 4. Launch Jupyter
jupyter notebook
```

Then open any `SE##_CA_*.ipynb` notebook and run the first setup cell.

---

## Workshop Schedule 🧠

### Day 1 – 23 February (Sessions 1–3)

| Time            | Activity                                         | Notebook |
| --------------- | ------------------------------------------------ | -------- |
| 08:30           | Registration, setup & troubleshooting            | |
| 09:00 – 10:15   | **Session 1:** Introduction to PyTorch           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CLDiego/uom_fse_dl_workshop/blob/main/SE01_CA_Intro_to_pytorch.ipynb) |
| 10:15 – 10:30   | Break                                            | |
| 10:30 – 11:45   | **Session 2:** Artificial Neural Networks        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CLDiego/uom_fse_dl_workshop/blob/main/SE02_CA_Artificial_neural_networks.ipynb) |
| 11:45 – 12:45   | Lunch                                            | |
| 12:45 – 14:00   | **Session 3:** Training Neural Networks          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CLDiego/uom_fse_dl_workshop/blob/main/SE03_CA_Training_neural_networks.ipynb) |

### Day 2 – 2 March (Sessions 4–5)

| Time            | Activity                                                  | Notebook |
| --------------- | --------------------------------------------------------- | -------- |
| 08:30           | Setup continuation                                        | |
| 09:00 – 10:30   | **Session 4:** Convolutional Neural Networks (CNNs)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CLDiego/uom_fse_dl_workshop/blob/main/SE04_CA_Convolutional_Neural_Networks.ipynb) |
| 10:30 – 10:45   | Break                                                     | |
| 10:45 – 12:00   | **Session 5:** Transfer Learning & Image Segmentation     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CLDiego/uom_fse_dl_workshop/blob/main/SE05_CA_Transfer_Learning.ipynb) |
| 12:00 – 13:00   | Lunch                                                     | |
| 13:00 – 14:00   | Wrap-up, discussion & Q&A                                 | |

> Light refreshments and lunch will be provided on both days.

---

## Learning Outcomes 🎯

By the end of the workshop, you'll be able to:

* Implement deep learning models using PyTorch
* Build and train artificial neural networks and autoencoders
* Apply CNNs for image classification tasks
* Use transfer learning with pre-trained models for image segmentation
* Work with real-world datasets for classification, regression, and anomaly detection tasks
* Understand data preprocessing, augmentation, and normalisation techniques

---

## Datasets 📊

You'll work with the following datasets across the sessions:

| Dataset | Task |
| ------- | ---- |
| **Higgs Boson Dataset** | Binary classification with high-energy physics data |
| **Heart & Lung Sounds (HLS-CMDS)** | Anomaly detection using autoencoders |
| **Historical Concrete Crack Dataset** | CNN-based image classification |
| **ISIC Skin Lesion Dataset** | Medical image segmentation using U-Net and transfer learning |

---

## Repository Structure 📁

```
UoM_fse_dl_workshop/
├── SE##_CA_*.ipynb      # Code-along notebooks for live exercises
├── solutions/           # Completed notebooks with full implementations
├── figs/                # Figures and diagrams
└── utils/               # Helper tools used throughout the workshop
    ├── plotting/
    ├── data/
    ├── ml/
    └── solutions.json
```

---

## Using the Exercise Checker ✅

Throughout the notebooks, you’ll find 🎯 exercises. Use the built-in checker to validate your answers.

```python
answer = {'your_solution': result}
checker.check_exercise(1, answer)
```

### Requesting Hints 💡

```python
checker.display_hints(1)
```

✔️ Correct = green check
❌ Incorrect = feedback provided
💬 Hints are tailored to the task

---

## Common Workflows

1. Read the exercise and implement the solution.
2. Use the checker to validate your work.
3. Request hints if needed.
4. Learn from any mistakes and try again.

---

## Prerequisites 📾

* Basic Python programming skills
* Familiarity with Jupyter Notebooks, NumPy, and Pandas
* Understanding of core ML concepts (e.g., loss functions, model evaluation)
* No PyTorch experience required!
* (Optional) Background in linear algebra and calculus is beneficial

---

## Additional Resources 📚

### PyTorch & Models

* [PyTorch Docs](https://pytorch.org/docs/stable/)
* [torchvision Models](https://pytorch.org/vision/stable/models.html)
* [PyTorch Hub](https://pytorch.org/hub/)
* [Hugging Face Models](https://huggingface.co/models)

### Visual Tools

* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [Distill Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [TensorBoard for PyTorch](https://pytorch.org/docs/stable/tensorboard.html)

### Courses & Books

* [Deep Learning with PyTorch (60-min Blitz)](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [Fast.ai Course](https://course.fast.ai/)
* [Dive into Deep Learning](https://d2l.ai/)
