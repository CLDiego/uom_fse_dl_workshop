![](https://i.imgur.com/mTltzAc.png)

> **⚠️ Development Branch Notice**: This `colab` branch is primarily for development purposes. For the stable version of the workshop materials, please refer to the `main` branch.

## Workshop Overview 📋

This hands-on workshop will guide participants through the fundamentals of deep learning with PyTorch, covering:

- PyTorch's tensor operations and automatic differentiation
- Building and training neural networks for various tasks
- Implementing CNNs for computer vision
- Applying transfer learning with pre-trained models
- Working with sequential data
- Developing multi-class classification models

## Setting Up Your Development Environment 🛠️

### Recommended IDE: Visual Studio Code

We recommend using Visual Studio Code (VS Code) for this workshop as it provides excellent Python and Jupyter Notebook support.

#### Installing VS Code
1. Download VS Code from the [official website](https://code.visualstudio.com/).
2. Follow the installation instructions for your operating system.
3. Install the following extensions for a better experience:
   - Python
   - Jupyter
   - Pylance
   - Python Indent
   - IntelliCode

To install extensions in VS Code, click on the Extensions icon in the sidebar (or press `Ctrl+Shift+X`), search for each extension, and click "Install".

### Why Use a Virtual Environment? 🔒

Using a virtual environment is essential for Python development because it:

- Isolates project dependencies, preventing conflicts between projects
- Makes your project reproducible across different machines
- Allows easy management of package versions
- Keeps your global Python installation clean

### Setting Up a Virtual Environment 📦

#### Windows

```powershell
# Create a virtual environment
python -m venv dl_workshop_env

# Activate the environment
dl_workshop_env\Scripts\activate

# Deactivate the environment when finished
deactivate
```

#### Linux/macOS

```bash
# Create a virtual environment
python -m venv dl_workshop_env

# Activate the environment
source dl_workshop_env/bin/activate

# Deactivate the environment when finished
deactivate
```

### Required Libraries 📚

Install the necessary packages after activating your virtual environment:

```bash
pip install matplotlib numpy pandas scikit-learn jupyter ipywidgets ipympl ipykernel urllib3 requests tqdm albumentations networkx scipy
```

For GPU support (NVIDIA GPUs only):

```bash
# Install the latest version of PyTorch with CUDA support

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Note: Replace 'cu118' with the appropriate CUDA version for your system.
```

No GPU? No problem! PyTorch will automatically fall back to CPU. Install the CPU version of PyTorch using the command below:

```bash
pip install torch torchvision torchaudio
```

## Workshop Content 📖

### Session 1: Introduction to PyTorch (~0.5 hours)
- PyTorch basics and tensors
- Creating and manipulating tensors
- Indexing and slicing tensors
- Tensor operations (arithmetic, matrix operations)
- Broadcasting
- Reshaping tensors
- Automatic differentiation with autograd
- Working with data in PyTorch
- Using GPUs for acceleration

### Session 2: Artificial Neural Networks (~1 hour)
- Understanding neurons and perceptrons
- Forward and backward pass
- Neural network components
- Case study: Space Shuttle Challenger disaster
- Implementing a simple neural network from scratch
- Training neural networks
- Evaluation and prediction

### Session 3: Training Neural Networks (~1.5 hours)
- PyTorch workflow for training models
- Working with the ARKOMA robotics dataset / PDE with Physics Informed Neural Networks
- Data preparation and normalization
- Activation functions and their purposes
- Model architecture design
- Weight initialization techniques
- Choosing optimizers and loss functions
- Creating effective training loops
- Handling overfitting and early stopping
- Model evaluation techniques

### Session 4: Convolutional Neural Networks (~1.5 hours)
- CNN fundamentals and operations
- Convolution basics and filter design
- Implementing custom filters
- Image data preparation with transforms
- Building CNN models from scratch
- Historical Crack Dataset exploration
- Using pooling layers
- Regularization techniques
- CNN architectures (SimpleNet and TinyVGG)
- Training and evaluating CNN models

### Session 5: Transfer Learning (~1.5 hours)
- Transfer learning fundamentals
- Medical image segmentation with ISIC dataset
- Data preparation for segmentation tasks
- U-Net architecture implementation
- Segmentation-specific loss functions (Dice loss)
- Using pre-trained models as feature extractors
- EfficientNet-based U-Net implementation
- Comparing models trained from scratch vs. transfer learning

## Intended Learning Outcomes (ILOs) 🎯

By the end of the workshop, participants will be able to:
- Implement deep learning models using PyTorch for diverse applications
- Apply CNNs for image classification and segmentation tasks
- Utilize transfer learning to adapt pre-trained models to specific problems
- Work with real-world datasets to solve classification and regression problems

## Workshop Materials Structure 📁

### Repository Organization
```
UoM_fse_dl_workshop/
├── datasets/                  # Dataset storage
├── solutions/                 # Solved notebooks
├── figs/                      # Workshop images and icons
├── utils/                     # Utility functions and checkers
│   ├── core.py                # Exercise checker implementation
│   ├── data.py                # Data loading utilities
│   ├── ml/                    # ML helper functions
│   └── solutions.json         # Exercise solutions and hints
└── SE01_CA_Intro_to_pytorch.ipynb    # Workshop notebooks
```

### Using the Exercise Checker 📝

Each workshop notebook contains exercises marked with 🎯. The exercise checker helps you verify your solutions and provides hints when needed.

#### Checking Your Solutions
```python
# At the end of each exercise, there will be a check block:
answer = {
    'your_solution': your_tensor,
    'another_part': another_result
}
checker.check_exercise(exercise_number, answer)
```

#### Getting Hints
If you're stuck, you can get hints for any exercise:
```python
# Display hints for exercise 1
checker.display_hints(1)
```

#### Exercise Feedback
The checker provides immediate feedback:
- ✅ Correct solutions are marked with a green checkmark
- ❌ Incorrect solutions show what went wrong
- 💡 Helpful hints appear when needed

### Common Workflows

1. **Starting an Exercise**:
   - Read the exercise description carefully
   - Implement your solution in the provided code cell
   - Run the cell to see if it works

2. **Checking Your Work**:
   - The checker automatically validates your solution
   - Multiple aspects may be checked (values, shapes, types)
   - All parts must be correct to pass

3. **Getting Help**:
   - Use `checker.display_hints()` for guidance
   - Hints are context-aware and specific to each exercise
   - Multiple hints may be available per exercise

4. **Learning from Mistakes**:
   - Pay attention to error messages
   - Check tensor shapes and types when debugging
   - Use print statements to understand intermediate results

## Getting Started 🚀

1. Clone this repository
2. Set up a virtual environment as described above
3. Install the required dependencies
4. Open the Jupyter notebooks in the repository to follow along with the workshop materials

## Support 💬

For issues or questions during the workshop, please reach out to the workshop instructors.

## Prerequisites 📝

- Basic Python programming knowledge
- Familiarity with fundamental machine learning concepts
- A laptop with Python 3.8+ installed
- Basic understanding of linear algebra and calculus (optional but helpful)
- Familiarity with Jupyter Notebooks (optional but helpful)

## Additional Resources 📚

To further enhance your learning experience, here are some valuable external resources:

### PyTorch Documentation

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html) - Comprehensive API reference and tutorials
- [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html) - Quick reference guide for common PyTorch operations
- [PyTorch Examples Repository](https://github.com/pytorch/examples) - Official examples for various deep learning tasks

### Pre-trained Models & Transfer Learning

- [torchvision.models](https://pytorch.org/vision/stable/models.html) - Documentation for pre-trained models in torchvision
- [PyTorch Hub](https://pytorch.org/hub/) - Repository of pre-trained models ready for fine-tuning
- [Hugging Face Models](https://huggingface.co/models) - Thousands of pre-trained models for various tasks

### Visual Learning Tools

- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive visualization of convolutional neural networks
- [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) - Understanding neural networks through feature visualization
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization toolkit compatible with PyTorch (via torch.utils.tensorboard)

### Tutorials & Courses

- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) - Quick PyTorch introductory tutorial
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official collection of tutorials from basic to advanced topics
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - Practical deep learning course using PyTorch

### Papers & Research

- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Original U-Net architecture paper for biomedical image segmentation
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Scaling up CNNs more efficiently
- [Transfer Learning Survey](https://arxiv.org/abs/1911.02685) - Comprehensive survey on transfer learning

### Books

- [Dive into Deep Learning](https://d2l.ai/) - Interactive book with code examples in multiple frameworks including PyTorch

## VS Code Setup Guide for Workshop Participants

To ensure all participants have a consistent development environment, please follow these steps to set up Visual Studio Code:

### 1. Install Visual Studio Code

Download and install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)

### 2. Install Required Extensions

Launch VS Code and install these extensions:

1. **Python**:
   - Press `Ctrl+Shift+X` to open the Extensions panel
   - Search for "Python"
   - Install the one by Microsoft (most downloaded)

2. **Jupyter**:
   - Search for "Jupyter"
   - Install the extension by Microsoft

3. **Pylance**:
   - Search for "Pylance"
   - Install Microsoft's Pylance for enhanced Python language support

4. **Python Indent**:
   - Search for "Python Indent"
   - Install to automatically fix Python indentation

5. **IntelliCode**:
   - Search for "IntelliCode"
   - Install Microsoft's AI-assisted code completion tool

### 3. Configure Workspace Settings

Create workspace settings to ensure consistent behavior:

1. In your workshop directory, create a folder called `.vscode` if it doesn't exist
   - `mkdir .vscode`

2. Inside the `.vscode` folder, create a file called `settings.json`
   - Windows: `notepad .vscode\settings.json`
   - Mac/Linux: `touch .vscode/settings.json`

3. Copy these settings into the file:

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "autopep8",
    "editor.formatOnSave": true,
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "python.envFile": "${workspaceFolder}/.env",
    "editor.rulers": [88],
    "editor.renderWhitespace": "all",
    "python.linting.flake8Enabled": false,
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "files.trimTrailingWhitespace": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    "jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"]
}
```
4. For Windows users, add this line to the settings.json file:
```json
"python.defaultInterpreterPath": "${workspaceFolder}\\dl_workshop_env\\Scripts\\python.exe",
```
5. For Mac/Linux users, add this line to the settings.json file:
```json
"python.defaultInterpreterPath": "${workspaceFolder}/dl_workshop_env/bin/python",
```
6. Save the file and close it.

7. Configure Theme (Optional)
For better readability of code and notebooks:

- Press Ctrl+K Ctrl+T to open the theme selector
- Select "Dark+" or "Monokai" for good syntax highlighting

8. Additional VS Code Tips for This Workshop
- Open Workshop Folder: Use File > Open Folder... to open the entire workshop directory
- Explorer View: Use the Explorer view (Ctrl+Shift+E) to navigate between files
- Integrated Terminal: Use the integrated terminal (Ctrl+`) to run commands
- Interactive Window: Run Python code interactively with "Python: Create Interactive Window"
- Split Editor: Drag a tab to split the editor and view multiple files side by side
- Zen Mode: Use View > Appearance > Zen Mode (Ctrl+K Z) for distraction-free coding

### VS Code Shortcuts

- **Command Palette**: `Ctrl+Shift+P` (access all commands and settings)
- **Open Terminal**: `Ctrl+` (backtick) (open integrated terminal)
- **Run Cell**: `Shift+Enter` (execute the current cell in Jupyter Notebook)
- **Format Document**: `Shift+Alt+F` (format the current document)
- **Toggle Sidebar**: `Ctrl+B` (show/hide the sidebar)
- **Switch Tabs**: `Ctrl+Tab` (cycle through open tabs)
- **Search Files**: `Ctrl+P` (quickly open files by name)
- **Comment/Uncomment Line**: `Ctrl+/` (toggle comment on the current line or selection)

### Online tutorials for VS Code

- [VS Code Documentation](https://code.visualstudio.com/docs) - Official documentation for VS Code
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial) - Getting started with Python in VS Code
- [VS Code Jupyter Notebook Support](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) - Working with Jupyter Notebooks in VS Code
- [VS Code Extensions](https://code.visualstudio.com/docs/editor/extension-gallery) - Explore and install extensions for VS Code
- [VS Code Keyboard Shortcuts](https://code.visualstudio.com/docs/getstarted/keybindings) - List of keyboard shortcuts for VS Code
- [VS Code Themes](https://marketplace.visualstudio.com/search?target=VSCode&category=Themes) - Explore themes to customize the look of VS Code