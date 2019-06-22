# mma_2


Overview:<br />
<br />
The goal of this project is to build a good predictor of mma fights.<br />
<br />
Scraper collects as many mma fights as possible.<br /> 
<br />
Feature extractor collects features for every matchups. Current features collect a list of features from different fight subsets such as wins, losses and past n fights. These features include elo, win rate, number of fights and number of different types of finishes. The feature selector stores the data of 250k fights in pickled dataframes.<br />
<br />
Model manager reads the pickled dataframes and extract the feature arrays. This is fed into various ensemble models. I have seen best results (65-66%) with gradient boosting models especially catboost. <br />
<br />
Implemented improvements:<br />
-Adding multiple elos improved accuracy from 64.5 to 65.5%. This allows the features to store information with different levels of streakiness.<br />
-tested using the raw data for fighter past fights vs comparing absolute and percentage differences. Having both the raw values and the differences had the best results and improved accuracy.<br />
<br />
<br />
Insights:<br />
In the decision tree below, Elo splits first, then win rates:<br />
![dtree image](https://raw.githubusercontent.com/tristan00/mma_2/master/images/dtree2.PNG "Description goes here")<br />
<br />
Elo 2-8 is the traditional elo formula with increasing values of k. Elo 8 or the highest k elos dominate. This would imply that streaks happen frequently as elo is lost rapidly over a couple losses.<br />
Many of these elo features are understated as there are duplicate values from different fight subsets.<br />
![dtree image](https://raw.githubusercontent.com/tristan00/mma_2/master/images/feature_importance.PNG)<br />
<br />

TO DO:<br /> 
-Hyperparameter grid search.<br />
-Add location data to quantify homegame vs away game advantage.<br />
-add unique fight ids to allow for rare cases of multiple fights per night per fighter.<br />
-compare adding meta features at layer 1 vs layer 2<br />
-clean up and comment code in finished parts<br />
-Add diference and dvision of first layer outputs in stacked models.<br />
