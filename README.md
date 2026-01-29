# 25-26-Coding-Five-Approaches-to-Machine-Intelligence
Task 1: 


DataSets: 
https://www.kaggle.com/datasets/salahuddinahmedshuvo/grocery-inventory-and-sales-dataset https://huggingface.co/datasets/AmirMohseni/GroceryList https://huggingface.co/datasets/infinite-dataset-hub/GroceryItemClassification




Vocabulary Expansion (Recall): 
* The Kaggle dataset provided a solid baseline of common inventory (e.g., "Milk", "Bread"), but the Hugging Face datasets introduced critical variations and specific brand names (e.g., "Almond Milk", "Sourdough").
* Rationale: By aggregating these sources, the system’s vocabulary was significantly expanded. This directly reduced the "Unknown Category" error rate during testing, as the model was exposed to a wider variety of synonyms and specific product types.
* Training the LinearSVC on this diverse data ensured the model learned generalised linguistic patterns (e.g., associating the word "Crunch" with Cereal or Snacks) rather than memorising a specific, limited list from a single author.
* Source 1 was heavily skewed towards fresh produce. Integrating Source 3 (Infinite Dataset) provided necessary examples for under-represented categories like Household Chemicals and Pet Food, ensuring the classifier did not become biased toward only predicting food items.
This resulted in  1314 items being used. 
2. Rationale for Machine Learning Technique
Selected Approach: Linear Support Vector Machine (LinearSVC) with TF-IDF
To solve this text classification problem, I used Linear Support Vector Machine (SVM and TF-IDF.
Why LinearSVC and TF-IDF?
* TF-IDF (Term Frequency-Inverse Document Frequency): This transforms raw text into numerical data. Unlike simple counting, it down-weights common words (like "the" or "pack") and highlights discriminatory terms (like "Cheddar" or "Sirloin"), which helps the model focus on the most meaningful keywords.
* LinearSVC: For short-text classification with high-dimensional sparse data, LinearSVC is computationally efficient which is good  use for quick answers especially on my laptop. 
4. Analysis of Results
4.1 Generalisation Capabilities
The most significant result observed was the model's ability to classify unseen data.
* Observation: When testing with the input "Sour Apple" (it is not present in the training data), the model correctly predicted "Fruits & Vegetables". This confirms that the model is not memorising strings (overfitting) but has learned that the token "Apple" is a dominant feature for the Fruit category. A rule-based system looking for an exact match would have failed here.
4.2 Handling Ambiguity
The model demonstrated interesting behavior with ambiguous terms.
* Observation: The input "Almond Milk" was classified as "Dairy".
* Even though almonds give nuts and milk gives dairy. It's able to place almond milk as a Dairy. Technically, almond milk is a  pantry item, but commonly it is often grouped with dairy. The model likely learned this association because the word "Milk" has a massive TF-IDF weight for the Dairy category. 
________________


Task 2 
Engine
1. Introduction


Datasets used: Primary Database Chosen: Groceries Dataset (Kaggle)
Secondary Reference: Basic Dataset (GitHub)
Composition:
The model relies primarily on the Kaggle dataset, which contains approximately 38,765 rows of transactional data. And a Github DS.
1. The GitHub dataset (groceries - groceries.csv) was evaluated but ultimately treated as a secondary source as for its unreliability and its does provide nearly 10k additional pieces of data to fill in the gaps of the Kaggle DB
2. Statistical Significance: With nearly 40,000 entries, the Kaggle dataset provided enough volume to filter out noise. This meant I had to add a  Frequency Threshold while still retaining a large enough vocabulary to make interesting recommendations, avoiding the "Small Sample Size" bias found in smaller datasets.


2. Rationale for ML Technique
Selected Approach: Frequency-Based Association Rules
I chose a Frequency-Based Scoring System
Why this Technique?
1. Interpretability: Unlike complex Neural Networks ("black boxes"), this method is transparent. It relies on counting actual co-occurrences. If the system recommends "Rolls" with "Sausage," it is because the data explicitly shows they appear together frequently.
2. Reliability: This approach calculates the Confidence metric , which represents the conditional probability that Item B is in the basket given Item A is present. This is a standard statistical measure for discovering relationships in unsupervised data. Giving it a score which can be used for the model.
4. Analysis of Results
4.1 Whole Milk 
A significant result observed during testing was the dominance of "Whole Milk" in the recommendations.
* Result: Inputting diverse items like "Sausage," "Yogurt," or "Vegetables" often resulted in a recommendation for "Whole Milk."
* Explanation: This result is due to the Confidence metric used. Since "Whole Milk" is the most popular item in the entire dataset (appearing in roughly 45% of all transactions), it statistically has a high probability of appearing in any basket, regardless of the other items.
* Implication: While mathematically correct (people do buy milk with everything), this highlights a limitation of simple frequency-based ML: it favors globally popular items over niche, specific pairings.
4.2 Strong Associations
Despite the popularity bias, the model successfully identified specific strong pairs:
* Input: "Ham" ->  Output: "White Bread"
* Input: "Berries" ->  Output: "Yogurt"
* Analysis: In these cases, the specific connection between the two items was strong enough to outrank the generic popularity of milk. This validates that the Association Rule logic is working correctly to identify complementary goods.












________________
Task 3 
1. Introduction
DataSet: https://www.kaggle.com/datasets/liamboyd1/multi-class-food-image-dataset
* Produce: Apple, Banana, Cucumber, Tomato
* Dairy: Milk, Cheese, Butter
* Meat: Chicken, Bacon 
* This was used as it was all my laptop could handle 
Structural Compatibility: This dataset is pre-organised into labelled subdirectories (e.g., /train/apple, /train/banana). This folder structure is natively compatible with the TensorFlow image_dataset_from_directory function, allowing for efficient data ingestion without complex manual labelling.


Visual Diversity: The dataset provides high variance within each class. For example, the "Apple" folder contains images of red, green, and yellow apples, as well as whole and sliced fruit. This diversity is critical for training a Convolutional Neural Network (CNN) to learn generalised features (like shape and texture) rather than memorising specific colors or backgrounds.


Scalability: The images are of sufficient resolution to be resized to the target input shape of 180x180 pixels without significant loss of feature detail, ensuring the model receives clear visual data for feature extraction.


2. Rationale for Machine Learning Technique
Selected Approach: Convolutional Neural Network (CNN)
To solve the image classification problem, I implemented a Convolutional Neural Network (CNN) using TensorFlow and Keras. A CNN was chosen because of its property of Spatial Invariance and Filters: The CNN uses sliding filters to detect features like edges, textures, and curves regardless of where they appear in the image.
5. Conclusion
It worked better than expected. When Shown images that weren't in training data it gave a very solid guess. For example when shown an apple it guessed a tomato which is a very reasonable guess. It did however take about  2 hours to run the training data on my laptop. 
________________






How To Run: 
Task 1:
pip install kagglehub pandas datasets scikit-learn
Launch task1.py via Python Terminal




Task 2:
pip install kagglehub pandas datasets scikit-learn
Launch 2526 Coding Five - Task 2.py via Python Terminal




Task 3 
pip install tensorflow pillow numpy
Clone The Repository and ensure grocery_model.h5 and class_indices.txt are  present 
Launch 2526 Coding Five - Task 3.py via Python Terminal .#






Git Link: https://github.com/tnvr-cs/25-26-Coding-Five-Approaches-to-Machine-Intelligence/tree/main
