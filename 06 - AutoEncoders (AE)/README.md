# Movie Recommendation System Using Autoencoders

This project implements a **movie recommendation system** using  **Autoencoders (AE)** . The system is built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, which contains user ratings for movies. The goal is to predict user ratings for movies they haven't seen yet and recommend movies with the highest predicted ratings.

## Table of Contents

1. [Project Overview]()
2. [Dataset]()
3. [Model Architecture]()
4. [Implementation Details]()
5. [Results]()
6. [How to Run the Code]()
7. [Dependencies]()
8. [Future Improvements]()

---

## Project Overview

The goal of this project is to build a **movie recommendation system** using an  **Autoencoder (AE)** . The system learns to predict user ratings for movies by compressing and reconstructing the input data (user-movie ratings) through a neural network. The model is trained on the MovieLens dataset, and the predicted ratings are used to recommend movies to users.

---

## Dataset

The project uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, which contains:

* **User ratings** for movies (on a scale of 1 to 5).
* **Movie metadata** (e.g., title, genre).
* **User information** (e.g., user ID).

The dataset is preprocessed to create a user-movie matrix, where each row represents a user and each column represents a movie. The values in the matrix are the ratings given by users to movies. Missing values (unrated movies) are set to 0.

---

## Model Architecture

The recommendation system is built using a **Stacked Autoencoder (SAE)** with the following architecture:

### Encoder:

1. **Input Layer** : Takes the user-movie ratings as input (size = number of movies).
2. **Hidden Layer 1** : Fully connected layer with 20 neurons and a Sigmoid activation function.
3. **Hidden Layer 2** : Fully connected layer with 10 neurons and a Sigmoid activation function.

### Decoder:

1. **Hidden Layer 3** : Fully connected layer with 20 neurons and a Sigmoid activation function.
2. **Output Layer** : Fully connected layer with the same size as the input layer (number of movies). No activation function is applied here.

### Loss Function:

* **Mean Squared Error (MSE)** : Used to measure the difference between the predicted ratings and the actual ratings.

### Optimizer:

* **Stochastic Gradient Descent (SGD)** : Used to update the model's weights during training.

---

## Implementation Details

1. **Data Preprocessing** :

* The user-movie matrix is created from the MovieLens dataset.
* Missing ratings are set to 0.
* The dataset is split into training and test sets.

1. **Training** :

* The model is trained for 200 epochs.
* For each user, the model predicts ratings for all movies.
* The loss is computed only for movies that the user has rated.

1. **Evaluation** :

* The model's performance is evaluated on the test set using the RMSE (Root Mean Squared Error) metric.

1. **Recommendations** :

* After training, the model predicts ratings for movies that the user hasn't rated.
* Movies with the highest predicted ratings are recommended.

---

## Results

* The model achieves a low RMSE on the test set, indicating accurate predictions.
* Example recommendations are generated for users based on their predicted ratings.

---

## How to Run the Code

1. Clone the repository:

   ```
   git clone https://github.com/dash7ou/deep-learning.git
   cd deep-learning/06 - AutoEncoders (AE)
   ```
2. Install the required dependencies (see [Dependencies]()).
3. Download the MovieLens dataset from [here](https://grouplens.org/datasets/movielens/) and place it in the `data` folder.
4. Run the Jupyter Notebook or Python script:

   ```
   jupyter notebook Movie_Recommendation_System.ipynb
   ```
5. Follow the instructions in the notebook to train the model and generate recommendations.

---

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Matplotlib (for visualization)

Install the dependencies using:

```
pip install torch numpy pandas matplotlib
```

---

## Future Improvements

1. **Hyperparameter Tuning** :

* Experiment with different numbers of layers, neurons, and activation functions to improve model performance.

2. **Advanced Architectures** :

* Use **Variational Autoencoders (VAE)** or **Denoising Autoencoders (DAE)** for better recommendations.

3. **Cold Start Problem** :

* Incorporate additional user and movie metadata (e.g., genres, demographics) to handle new users or movies with no ratings.

4. **Deployment** :

* Deploy the model as a web application or API for real-time recommendations.

---

## Acknowledgments

* The [MovieLens](https://grouplens.org/datasets/movielens/) dataset is provided by GroupLens Research.
