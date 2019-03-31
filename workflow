. Essential libraries used: pandas, sklearn, nltk, xgboost.
Data preproccessing involved removing stopwords, stemming, identifying and removing outliers (using isolation forest).
Converted the output string data into integers using label encoder,
and created corpus containg all the bag of input words using CountVectorizer and fed this to Xgboost classifier.
