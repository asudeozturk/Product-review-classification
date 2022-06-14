# Product Review Classification
Multi-class sentiment analysis of e-commerce product reviews using Naive Bayes \
Natural Language Processing (course project)

## Dataset
Women's E-Commerce Clothing Reviews
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews \
Classes: Product ratings on a scale of 1 to 5

## Steps
1. Preprocessing
- Remove empty reviews
- Lowercasing
- Split dataset into training (80%) and test (20%) datasets
- Balance the training set by upsampling minority class
- Tokenize reviews
- Build vocab vector for the training set
2. Train the model
3. Test the model
4. Evaluate the model
- Confusion matrix graph
- ROC Curve graph
5. Ask user to enter a review and classify the input


## Findings 
Following results are observed after training and testing the model:

<figure>
  <img src="/img/confusion-matrix.png" width="400px" alt="confusion matrix graph" </img>  
  <img src="/img/roc-curve.png" width="400px" alt="ROC curve graph" </img> 
</figure>

## Demo
Example reviews and predicted ratings
<figure>
  <img src="/img/demo.png" width="400px" alt="user input example"
</figure>
