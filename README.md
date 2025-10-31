# CS315-Project-1
Decision Tree

We found that no database contains both pitch metric data and injury data. Thus, we used Baseball Savant for the pitch metrics data and Sportsradar for the injury data. 

Data handling problem: The pitch dataset includes pitch metrics from 2017 - current, while the injury data is only current injuries.

Potential solutions:
1) Finding a source of injury data from 2017 - current would give us more data to work with, and thus would be nice.
   -If it is possible to find year-by-year injury information, then it would be useful to split the pitch metric data year-by-year, and pair this with the injury data by year. This will give us features that are more      correlated to their respective output.
   -Possible source of injury data: https://github.com/robotallie/baseball-injuries
   
3) If solution 1 is not possible, then using pitch data from only the current season(possibly the season before as well) would give us data that better reflects circumstances that could predict injury.
   

Potential Solution to match pitch metrics to injury data:

The data from the CSV file has player name in the form: last_name, first_name. So, we should be able to split on the comma, flip the data to be [first_name, last_name], and match it to the first_name and last_name parameters from the API call data. That way, each player's pitch metrics and injury data are connected.

To gain a deeper understanding of Decision Trees, how they work, and how to implement Random Forest Regression,
I watched several Youtube videos from Normalized Nerd, including this tutorial on how to build a decision tree classifier from scratch:

Decision Trees Tutorial: https://www.youtube.com/watch?v=sgQAhG5Q7iY
Explanation of Decision Trees: https://www.youtube.com/watch?v=ZVR2Way4nwQ
Explanation of Random Forest: https://www.youtube.com/watch?v=v6VJ2RO66Ag&t=328s

