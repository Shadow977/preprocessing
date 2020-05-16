# Preprocessing script for the collected tweets

## Follow these steps to use the script:
Report if any bugs are found

### 1) Installation:
```bash
git clone https://github.com/Rutvik-Trivedi/preprocessing.git
pip install -r requirements.txt
```
After this, move the ```preprocessing.py``` file to the dataset folder ```Tweet-dataset```

### 2) Clean the dataset:
The following commands cleans the CSV file for the specified user:  
```python
from preprocessing import Cleaner
cleaner = Cleaner()
cleaner.clean_data('username.csv')  # Example: cleaner.clean_data('1capplegate.csv')
```
This will preprocess the particular CSV file and write the cleaned data to ```tweets/cleaned_data/username.csv```
It will remove the following things:
- Hashtags
- User Mentions
- StopWords
- Will lemmatize all the words. You may also try with Stemming
- Punctuations
- Emojis and all other junks
- URLs
- Replaces numbers with \<NUMBER\>
- Replaces words like 15k, 12cm, 100Km, etc with \<UNIT\>
- Processes words like can't, won't, I'll, etc

If you do not want to remove stop words, run the following:
```python
cleaner.clean_data('username.csv', remove_stopwords=False)
```
Similarly,
```python
cleaner.clean_data('username.csv', remove_hashtags=False) # Keep hashtags
cleaner.clean_data('username.csv', remove_mentions=False) # Keep @ mentions
```

You may add more to this script  

You may use the ```clean_data``` function in loop to loop through all users  

### 3) Analyse Data and plot:
The following commands may be used:
```python
from preprocessing import Analytics
analyser = Analytics()
max_word_counts, average = analyser.analyse_user('username.csv')
# Make sure you have cleaned the data first for the specific user

# Plot the data
analyser.plot_data('username.csv')
# This will save the png file of the bar graph to teets/users/username.png
# This way it will not show you the plot
```
If you just want to view the plot and not save it, use
```python
analyser.plot_data('username.csv', save=False)
```

### 4) Extract feature keywords:
You can now extract feature keywords from the document for each user using this script:  
dependency: ```scikit-learn```. If not installed, please run the following command again:
```bash
pip install -r requirements.txt
```
```python
from preprocessing import Analytics
analyser = Analytics()
top_keywords = analyser.extract_keywords('username.csv')
# By default, the top 5 keywords will be returned. If you want the top n keywords, run
top_n_keywords = analyser.extract_keywords('username.csv', max_features=n)  # Where n is an integer
```

### 5) Overall cleaning and analysis:
Run the following scripts:
```python
cleaner = Cleaner()
cleaner.clean()     # To clean all data
analyser = Analytics()
analyser.analyse()  # For overall analysis
```

#### You may contribute more to this if required.  
#### Please open an issue if any bug found
