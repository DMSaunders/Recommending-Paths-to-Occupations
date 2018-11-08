# Capstone Plan

## Research Questions, from easy to hard
* Is college major a good predictor of occupation code? Expecting no.
* What are the strongest predictors of occupation code given 15 'free will' features from the PUMS census?
* Of those features, which are most correlated with non-freewill features and what does that mean?
* Which of any of the features are good predictors of occupation code?
* What changes if I engineer my occupation target from occupation code, industry, other things?

## Initial Planning
- [] what are the assumptions about the data
    * I will find correlations between features and my target
    * Some of my 'free will' features will be correlated with nonfreewill demographics - address this
    * I will have enough people per the 23 occupation groups to treat them as separate datasets
    * Occupation codes are representative of reality. Since we know this is only partially true, cluster a new target?
- [] inputs/outputs
    * csv file for CA, unless there is not enough data, then csv for entire US
- get through the CRISP DM as fast as possible!! Then repeat

## Issues
* variablity of risk in careers (artist vs construction)
* people with 2 careers can only show one industry and occupation
* product is partial dependence plots, feature importance less important


## Scrum Meetings
* what I did yesterday
* today's plan
* biggest challenge

## Weds 14th Nov
MVP version of capstone due


## After Code Freeze: Mon 26th Nov
3-minute pitch practice
practice in class for 3-mins
capstone one-pager


## file structure
src
    features
    model
data
resources
notebooks
reports

## Project Names

* Free Will and Choice of Occupation
* Predictors of Occupation