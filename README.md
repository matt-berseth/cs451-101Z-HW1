# cs451-101Z-HW1
Repository for cs451 HW1 - Supervised Learning

## Windows 10/11 Set Up Steps
```powershell
# run the following in a powershell window

# clone the repo
git clone https://github.com/matt-berseth/cs451-101Z-HW1

# path into the repo
cd cs451-101Z-HW1

# create the virtual env
python3.10 -m venv .venv

# activate
.\.venv\Scripts\activate.ps1

# install the deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# launch vscode
code .
```

## Ubuntu 22.04 Set Up Steps
```bash
# run the following in a ubuntu window

# clone the repo
git clone https://github.com/matt-berseth/cs451-101Z-HW1

# path into the repo
cd cs451-101Z-HW1

# create the virtual env
python3 -m venv .venv

# activate
source ./.venv/bin/activate

# install the deps
pip install --upgrade pip
pip install -r requirements.txt

# launch vscode
code .
```

## Instructions
**Machine Learning Homework Assignment - Supervised Learning with MNIST**

**Objective:**  
The MNIST database contains handwritten digits and is a staple in the machine learning community. While deep learning methods have achieved impressive results on this dataset, traditional machine learning methods can still perform quite well. In this assignment, you'll use non-deep learning approaches to classify the digits in the MNIST dataset.

---

**Dataset:**  
The MNIST dataset. This dataset consists of 28x28 grayscale images of handwritten digits (0 through 9) and their corresponding labels. Each image is represented as a 784-dimension vector (28x28 pixels).

---

**Tasks:**

1. Complete the 4 tasks in the `main.py` python file.
1. Write a 1-2 page report summarizing the findings of tasks 3 and 4. Outline what model and corresponding hyper parameter settings do you believe create the best model. Provide evidence of why you believe this to be true. If you tried hyper parameter settings that did not work, include these results as well and provide insight into why you believed these attempts performed poorly. Do you think this model will perform well on data that it was not trained on? For task 4, include a description of what decisions/rules the model is using to choose a classification.

---

**Deliverables**:

1. Python code (`main.py`).
2. Report summarizing findings and insights (`.pdf`).

---

**Notes**:
- Avoid using deep learning or neural network-based approaches for this assignment.
- Plagiarism will result in a zero for the assignment. Always provide citations and references if you used any external resources.