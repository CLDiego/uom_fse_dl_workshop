![Workshop Banner](https://github.com/CLDiego/uom_fse_dl_workshop/blob/main/figs/banner.png)
<!-- [![GitHub Pages](https://img.shields.io/badge/View%20Site-GitHub%20Pages-blue?logo=github)](https://cldiego.github.io/uom_fse_dl_workshop/) -->

# Deep Learning with PyTorch – Workshop

## Overview 📋

This hands-on workshop introduces the fundamentals of deep learning using PyTorch. Participants will learn by building real models and solving practical tasks.

### What You’ll Learn

* Core PyTorch concepts (tensors, autograd, GPU usage)
* Building and training neural networks
* Creating Physics-Informed neural networks
* Implementing CNNs for vision tasks
* Applying transfer learning with pre-trained models
* Working with real-world datasets
* Designing classification and regression models

---

## Getting Started 🛠️

### ✅ Recommended Platform: [Google Colab](https://colab.research.google.com/)

Colab provides a free, GPU-enabled environment—ideal for this workshop.

#### What You Need

* A Google account
* Reliable internet connection

#### Running the Notebooks

1. Open the GitHub repo.
2. Click the “Open in Colab” badge (if available), or:

   * Download the notebook locally.
   * Open [Google Colab](https://colab.research.google.com/).
   * Use **File > Upload Notebook** to run it.

#### Enable GPU in Colab

1. **Runtime > Change runtime type**
2. Set **Hardware Accelerator** to `GPU`
3. Click **Save**

📘 [Colab Tips](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) | [Colab FAQ](https://research.google.com/colaboratory/faq.html)

#### Install Dependencies

Each notebook starts with a setup cell. Run it first to install all required libraries.

---

## Workshop Sessions 🧠

| Session          | Topic                                | Duration |
| ---------------- | ------------------------------------ | -------- |
| **1**            | PyTorch Basics & Tensors             | \~1 hr |
| **2**            | Artificial Neural Networks (ANNs)    | \~1.5 hr   |
| **3**            | Model Training & Optimization        | \~0.5 hr |
| **3B**           | Physics-Informed Neural Networks (PINNs) | \~1 hr |
| **4**            | Convolutional Neural Networks (CNNs) | \~2 hr |
| **5**            | Transfer Learning & U-Net            | \~2 hr |

---

## Learning Outcomes 🎯

By the end, you’ll be able to:

* Build and train models in PyTorch
* Apply CNNs to classification & segmentation
* Fine-tune pre-trained models on new tasks
* Use PyTorch effectively for real-world datasets

---

## Repository Structure 📁

```
UoM_fse_dl_workshop/
├── solutions/           # Completed notebooks
├── figs/                # Figures and diagrams
├── utils/               # Checker and data helpers
│   ├── plotting/
│   ├── data/
│   ├── ml/
│   └── solutions.json
└── SE01_CA_Intro_to_pytorch.ipynb   # Code-along notebooks
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

* Basic Python skills
* Some knowledge of basic machine learning concepts
* Familiarity with linear algebra/calculus (optional)
* No PyTorch experience required!

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
