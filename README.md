


## NLPSentimentAnalysis
Sentiment Analysis for Movie review.
### Description
Objective is to build machine learning algorithm to predict sentiment of review done on movie.
We have 25000 already tagged data, and model is train based on tagged data.


### Tools and Technique

- `Platform` :: python
- `Framework` ::  scikit learn
- `Technique` :: Vectorization (Matrix with different column for every disticnt word)
- `Algorithm` :: Logistic Regression

### Data

- `Train` :: 25000
- `Positive tagged data` ::  15000
- `Negetive tagged data` ::  15000
- `Test` :: 25000
- `Source` :: http://ai.stanford.edu/~amaas/data/sentiment/

### Files

- `SentimentAnalysisIMDB.py` :: Coding file for step by step process with details. 
- `finalized_model.sav` :: Saved model on local directory
- `finalized_model_cv.sav` :: Saved Count Vectorized representation of reviews.


### Conclusion

Model is train and tested with efficiency of 88% on test data.

### Top 5: Positive Word

('excellent', 1.365964583386036)
('refreshing', 1.2609655289082766)
('perfect', 1.2064915697312957)
('superb', 1.1376531892553046)
('appreciated', 1.1225944226055673)


### Top 5: Negetive Word

('worst', -2.0770743501386253)
('waste', -1.9167593125702842)
('disappointment', -1.6835827461330934)
('poorly', -1.6612995664050252)
('awful', -1.543518327212851)



