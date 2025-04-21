---

# **💡 Stroke Risk Prediction Project**

Welcome to the **Stroke Risk Prediction Project**! This project leverages machine learning techniques to predict the likelihood of stroke based on various health and demographic factors. The project is structured into multiple phases, each handled by a dedicated team member.

---

## **🚀 How to Run the Project**

### **Using Conda Environment**
1. **Create a Conda Environment**:
   ```bash
   conda create -n stroke-risk python=3.10 -y
   conda activate stroke-risk
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open code.ipynb in the Jupyter interface.

### **Using a Simple Python Environment**
1. **Set Up a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open code.ipynb in the Jupyter interface.

---

## **📂 Project Description**

### **Code Structure**
- **Data Preprocessing & Feature Engineering**: Handles missing values, encodes categorical variables, and scales data.
- **Exploratory Data Analysis (EDA)**: Visualizes data distributions, correlations, and outliers.
- **Model Training**: Trains a machine learning model to predict stroke risk.
- **Model Evaluation**: Evaluates the model using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

### **Dataset**
The dataset used in this project is the **Healthcare Stroke Dataset**, which contains features like age, gender, BMI, glucose levels, and more.

---

## **🛠️ Requirements**

### **For VS Code**
1. Install the Python extension for VS Code.
2. Install Jupyter extension for running `.ipynb` files.
3. Ensure the Python interpreter is set to the virtual/conda environment.

### **For PyCharm**
1. Install the Python plugin (if not already installed).
2. Configure the Python interpreter to point to the virtual/conda environment.
3. Install the Jupyter plugin for running notebooks.

---

## **🌟 How to Work with Branches**

This repository follows a **feature-branch workflow** to ensure smooth collaboration and code management. Below are the details of the existing branches and how to work with them:

### **Existing Branches**
1. **`feature/data-preprocessing-and-feature-engineering`**: Contains all code related to data preprocessing and feature engineering.
2. **`feature/eda`**: Contains all code related to exploratory data analysis.
3. **`feature/model-training`**: Contains all code related to model training.
4. **`feature/model-evaluation`**: Contains all code related to model evaluation.

### **Steps to Work with Branches**
1. **Switch to a Branch**:
   ```bash
   git checkout <branch-name>
   ```
   Example:
   ```bash
   git checkout feature/eda
   ```

2. **Make Changes**:
   - Add your code or make updates relevant to the branch's purpose.

3. **Commit Changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push Changes**:
   ```bash
   git push origin <branch-name>
   ```

5. **Create a Pull Request**:
   - Go to the repository on GitHub.
   - Click "New Pull Request" and select the branch you worked on.
   - Add a description of your changes and submit the pull request.

---

## **✨ Contribution Guidelines**

- **Main Branch**: Contains the stable version of the project.
- **Feature Branches**: Use feature-specific branches for development.
- **Pull Requests**: Ensure all changes are reviewed and approved before merging into the main branch.

---

Feel free to contribute to this project by forking the repository, creating new branches, and submitting pull requests. Let's work together to make this project a success! 🚀

---
