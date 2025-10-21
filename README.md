# CS315-Project-1
Decision Tree

We found that no database contains both pitch metric data and injury data. Thus, we used Baseball Savant for the pitch metrics data and Sportsradar for the injury data. 

Potential Solution to match pitch metrics to injury data:

The data from the CSV file has player name in the form: last_name, first_name. So, we should be able to split on the comma, flip the data to be [first_name, last_name], and match it to the first_name and last_name parameters from the API call data. That way, each player's pitch metrics and injury data are connected.

