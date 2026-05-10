# Face Recognition and Reconstruction via Singular Value Decomposition
This repository provides a comprehensive MATLAB implementation of a face recognition and image compression system utilizing **Singular Value Decomposition (SVD)**. The project, conducted on the Yale Face Database, explores the application of linear algebra to high-dimensional datasets, focusing on dimensionality reduction through Principal Component Analysis (PCA).

## 1. Abstract
The core objective of this study is to evaluate the efficacy of the **Eigenfaces** method for both identity classification and high-fidelity image reconstruction. By projecting facial data onto a lower-dimensional subspace, we demonstrate that a significant portion of a dataset's variance—often exceeding 90%—can be captured by a remarkably small number of singular vectors. This implementation addresses critical challenges in biometric data analysis, such as managing varying lighting conditions and the detection of unauthorized subjects through statistical thresholding.

## 2. Methodology and Implementation
The system follows a rigorous mathematical pipeline. All images are initially vectorized and centered against a global mean face to ensure that the SVD operates on the dataset's covariance matrix. The resulting left singular vectors, or "Eigenfaces," define the orthonormal basis of our facial subspace. 

Identification is achieved via a **Nearest Neighbor (NN)** classifier within the reduced space, using Euclidean distance to measure similarity between the projected test samples and the training database.

## 3. Key Findings
Experimental results reveal a distinct gap between the requirements for recognition and reconstruction. 
* **Recognition:** The system achieves peak accuracy for most expressions using as few as 20 features.
* **Reconstruction:** Visual output remains blurred at low ranks, requiring a significantly larger subspace to recover high-frequency textures and individual details.

Statistical reliability was verified through **Leave-k-Out Cross-Validation**, showing that the model's predictive power remains stable even as training data is reduced. Furthermore, by analyzing the distribution of distances, a rejection threshold ($\tau \approx 400$) was established to successfully mitigate the **False Acceptance Rate (FAR)** when presented with subjects outside the training set.

## 4. Repository Structure
* `accuracy_analysis.m`: Evaluates the relationship between the number of singular values and classification accuracy.
* `face_reconstruction.m`: Visualizes the progressive reconstruction of images across different ranks ($p$).
* `cross_validation.m`: Implementation of the Leave-k-Out statistical testing.
* `eigenfaces_visualizer.m`: Generates and displays the primary principal components.
* `docs/report.pdf`: Full academic report with mathematical proofs and detailed results.

---
*Developed as part of the Scientific Computing course.*
