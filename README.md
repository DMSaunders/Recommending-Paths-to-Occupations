# Predicting Occupation Based on Life Decisions: 
## So You Want to Be a Programmer? 

## What are the strongest predictors of occupation code given 16 'free will' factors from the 2017 US Census? 
Such as:
College Major
Educational Attainment
Location
Military Service

Young adults could enter a desired occupation into a web app and see its most predictive features as a suggestion of paths others have successfully taken. The site would explore correlations between these controllable predictors and uncontrollable demographics like wealth and emphasize that the best way to change entrenched patterns is to know what you have the power to change.

## Tech:
Coded in Python including Pandas and scikit-learn. Challenged by runtime while gridsearching models, minimal tuning gains, and feature selection. Learned to run pipelines and fire up a model in Jupyter on an AWS EC2 instance in 7 minutes. 

## Results:
I focused on predicting Computer Science and Math occupations. Of 8 different classification models, tree-based learning scored best across metrics. Although I found the majority of predictive power in the education variables of undergraduate major and educational attainment, I can show young adults how location is nearly as important as having a CS degree. Travel time to work, number of hours and weeks worked, working in for-profit sector, and English fluency matter, with caveats that the features work together and may be proxies for latent trends. As a next step I will improve dimensionality reduction, engineer more features, and create an only-women model which can take into account features like children which could not be part of the current dataset since it would cause leakage of gender as a feature. 
