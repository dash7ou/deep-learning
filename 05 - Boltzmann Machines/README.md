# Boltzmann Machines for Movie Recommendations

This project demonstrates the application of  **Boltzmann Machines** , specifically  **Restricted Boltzmann Machines (RBMs)** , in building a movie recommendation system using the [MovieLens](https://grouplens.org/datasets/movielens/) dataset.

## 📁 Project Structure

```
05 - Boltzmann Machines (BM)/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── users.csv
├── models/
│   ├── rbm.py
│   └── train.py
├── notebooks/
│   ├── RBM_Movie_Recommendations.ipynb
│   └── Data_Preprocessing.ipynb
└── README.md
```

* **data/** : Contains the MovieLens dataset files.
* **models/** : Includes the RBM model implementation and training scripts.
* **notebooks/** : Jupyter notebooks for data preprocessing and model training.

## 🎬 Dataset

The [MovieLens](https://grouplens.org/datasets/movielens/) dataset is a widely-used benchmark for evaluating recommendation systems. It comprises user ratings for various movies, facilitating collaborative filtering approaches.In this project, we utilize the MovieLens 100K dataset, which includes:

* **100,000 ratings** (ranging from 1 to 5 stars)
* **943 users**
* **1,682 movies**

Each user has rated at least 20 movies, ensuring sufficient data for training the recommendation model.

## 🧠 Understanding Boltzmann Machines

**Boltzmann Machines** are a type of stochastic recurrent neural network capable of learning probability distributions over their set of inputs. They are particularly useful for uncovering complex patterns in data.

A **Restricted Boltzmann Machine (RBM)** is a simplified version where the network is divided into two layers:

* **Visible Layer** : Represents the input data.
* **Hidden Layer** : Captures dependencies and patterns in the data.

In RBMs, there are no connections between units within the same layer, simplifying the training process.

## 🛠️ Implementation Details

1. **Data Preprocessing** :

* Load and preprocess the MovieLens dataset.
* Normalize ratings and split the data into training and testing sets.

1. **Model Architecture** :

* Define the RBM with a specified number of hidden units.
* Utilize binary units for both visible and hidden layers.

1. **Training** :

* Train the RBM using Contrastive Divergence, a common optimization algorithm for RBMs.
* Monitor reconstruction error to assess model performance.

1. **Evaluation** :

* Generate movie recommendations for users based on their rating history.
* Evaluate the model's accuracy using metrics such as Root Mean Squared Error (RMSE).

## 🚀 Getting Started

1. **Clone the Repository** :

```bash
   git clone https://github.com/dash7ou/deep-learning.git
   cd deep-learning/05\ -\ Boltzmann\ Machines\ \(BM\)
   ```

```

1. **Set Up the Environment** :

* It's recommended to use a virtual environment to manage dependencies.
* Install the required packages using the provided `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

  ```

1. **Run the Jupyter Notebooks** :

* Launch Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
  ```
* Open and execute the notebooks in the `notebooks/` directory to preprocess data and train the RBM model.

## 📊 Results

After training, the RBM model can predict user preferences and recommend movies that align with their tastes. The effectiveness of the recommendations can be evaluated using metrics like RMSE, with lower values indicating better performance.

## 🤝 Contributions

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## 📧 Contact

For any questions or inquiries, feel free to reach out:

* **GitHub** : [dash7ou](https://github.com/dash7ou)
* **LinkedIn** : [Mohammed M R Zourob](https://www.linkedin.com/in/mohammed-zourob-b9796819a)

Happy Coding!
