# CS315-Project-1
Decision Tree

We found that no database contains both pitch metric data and injury data. Thus, we used Baseball Savant for the pitch metrics data and Sportsradar for the injury data. 

Potential Solution to match pitch metrics to injury data:

The data from the CSV file has player name in the form: last_name, first_name. So, we should be able to split on the comma, flip the data to be [first_name, last_name], and match it to the first_name and last_name parameters from the API call data. That way, each player's pitch metrics and injury data are connected.

To gain a deeper understanding of Decision Trees, how they work, and how to implement Random Forest Regression,
I watched several Youtube videos from Normalized Nerd, including this tutorial on how to build a decision tree classifier from scratch:

Decision Trees Tutorial: https://www.youtube.com/watch?v=sgQAhG5Q7iY
Explanation of Decision Trees: https://www.youtube.com/watch?v=ZVR2Way4nwQ
Explanation of Random Forest: https://www.youtube.com/watch?v=v6VJ2RO66Ag&t=328s

