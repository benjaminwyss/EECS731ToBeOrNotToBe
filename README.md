# EECS 731 Project 2: To Be Or Not To Be
Submission by Benjamin Wyss

## Project Overview

Examining Shakespeare play data to build a classification model that predicts the player who speaks a specific line

### Data Set Used

All of Shakespeare's plays, characters, lines, and acts: https://www.kaggle.com/kingburrito666/shakespeare-plays

### Results

By utilizing feature engineering, I was able to build a decision tree based classification model which predicted the correct player who speaks a specific line with an accuracy of 83.59% in validation testing. This model worked by exploiting the observation that an finding a training sample with an identical play, act, scene, and player line number indicates an identical player who speaks a line. I attempted to further improve this model by performing word frequency analysis on players' spoken lines, but this analysis only complicated and tainted the trained models and was thus discarded in the final model. My full process and results are documented in the notebooks folder of this project. 
