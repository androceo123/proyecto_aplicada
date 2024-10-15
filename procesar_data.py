import pandas as pd
import re
import csv
from afinn import Afinn
import numpy as np
import skfuzzy as fuzz

# Load the data
file_path = 'test_data.csv'
data = pd.read_csv(file_path)
afinn = Afinn()

# Define a dictionary for common English abbreviations with variations
abbreviations = {
    "i'm": "i am", "im": "i am", "i m": "i am", "you're": "you are", "youre": "you are", "you re": "you are", 
    "he's": "he is", "hes": "he is", "he s": "he is", "she's": "she is", "shes": "she is", "she s": "she is", 
    "it's": "it is", "it s": "it is", "we're": "we are", "were": "we are", "we re": "we are", 
    "they're": "they are", "theyre": "they are", "they re": "they are", "i've": "i have", "ive": "i have", 
    "i ve": "i have", "you've": "you have", "youve": "you have", "you ve": "you have", "we've": "we have", 
    "weve": "we have", "we ve": "we have", "they've": "they have", "theyve": "they have", "they ve": "they have", 
    "i'll": "i will", "ill": "i will", "i ll": "i will", "you'll": "you will", "youll": "you will", "you ll": "you will", 
    "he'll": "he will", "hell": "he will", "he ll": "he will", "she'll": "she will", "shell": "she will", 
    "she ll": "she will", "we'll": "we will", "well": "we will", "we ll": "we will", "they'll": "they will", 
    "theyll": "they will", "they ll": "they will", "can't": "cannot", "cant": "cannot", "can t": "cannot", 
    "won't": "will not", "wont": "will not", "won t": "will not", "don't": "do not", "dont": "do not", 
    "do nt": "do not", "doesn't": "does not", "doesnt": "does not", "does nt": "does not", "didn't": "did not", 
    "didnt": "did not", "did nt": "did not", "isn't": "is not", "isnt": "is not", "is nt": "is not", 
    "aren't": "are not", "arent": "are not", "are nt": "are not", "wasn't": "was not", "wasnt": "was not", 
    "was nt": "was not", "weren't": "were not", "werent": "were not", "were nt": "were not", 
    "hasn't": "has not", "hasnt": "has not", "has nt": "has not", "haven't": "have not", "havent": "have not", 
    "have nt": "have not", "hadn't": "had not", "hadnt": "had not", "had nt": "had not", 
    "wouldn't": "would not", "wouldnt": "would not", "would nt": "would not", 
    "shouldn't": "should not", "shouldnt": "should not", "should nt": "should not", 
    "couldn't": "could not", "couldnt": "could not", "could nt": "could not", 
    "mustn't": "must not", "mustnt": "must not", "must nt": "must not", 
    "let's": "let us", "lets": "let us", "let s": "let us", 
    "that's": "that is", "thats": "that is", "that s": "that is", 
    "who's": "who is", "whos": "who is", "who s": "who is", 
    "what's": "what is", "whats": "what is", "what s": "what is", 
    "here's": "here is", "heres": "here is", "here s": "here is", 
    "there's": "there is", "theres": "there is", "there s": "there is", 
    "where's": "where is", "wheres": "where is", "where s": "where is", 
    "how's": "how is", "hows": "how is", "how s": "how is", 
    "y'all": "you all"
}

# Function to clean the text
def clean_text(text):
    # Expand abbreviations (with and without apostrophes and spaces)
    for abbr, full_form in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full_form, text, flags=re.IGNORECASE)

    # Handle the special case for "its" as a possessive pronoun (ignore if followed by a noun or adjective)
    text = re.sub(r"\bits\b(?!\s+(own|[a-zA-Z]+))", "it is", text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase and remove non-ASCII characters
    text = text.lower().encode('ascii', 'ignore').decode()
    
    return text

# Apply the cleaning function to the 'sentence' column
data['sentence'] = data['sentence'].apply(clean_text)

# Calculate sentiment scores for each sentence based on the AFINN lexicon
# Positive score: sum of positive word scores
data['positive_score'] = data['sentence'].apply(lambda x: sum([afinn.score(word) for word in x.split() if afinn.score(word) > 0]))

# Negative score: sum of negative word scores
data['negative_score'] = data['sentence'].apply(lambda x: sum([afinn.score(word) for word in x.split() if afinn.score(word) < 0]))

# Fuzzification of the scores

# Define the fuzzy membership functions for positive and negative scores
positive_max = data['positive_score'].max()
negative_max = abs(data['negative_score'].min())

x_positive = np.linspace(0, positive_max, 100)
x_negative = np.linspace(0, negative_max, 100)

positive_low = fuzz.trimf(x_positive, [0, 0, positive_max/2])
positive_medium = fuzz.trimf(x_positive, [0, positive_max/2, positive_max])
positive_high = fuzz.trimf(x_positive, [positive_max/2, positive_max, positive_max])

negative_low = fuzz.trimf(x_negative, [0, 0, negative_max/2])
negative_medium = fuzz.trimf(x_negative, [0, negative_max/2, negative_max])
negative_high = fuzz.trimf(x_negative, [negative_max/2, negative_max, negative_max])

# Function to fuzzify a value based on membership functions
def fuzzify(value, membership_functions, value_range):
    memberships = [fuzz.interp_membership(value_range, mf, value) for mf in membership_functions]
    return [float(m) for m in memberships] 

# Fuzzify the positive and negative sentiment scores
data['positive_fuzzy'] = data['positive_score'].apply(lambda x: fuzzify(x, [positive_low, positive_medium, positive_high], x_positive))
data['negative_fuzzy'] = data['negative_score'].apply(lambda x: fuzzify(abs(x), [negative_low, negative_medium, negative_high], x_negative))

# Save the dataset with fuzzified sentiment scores to a new CSV file
output_file_path = 'tweets_with_fuzzy_scores.csv'
data[['sentence', 'positive_score', 'negative_score', 'positive_fuzzy', 'negative_fuzzy']].to_csv(output_file_path, index=False, quoting=csv.QUOTE_ALL)

print(f"Processed file saved at {output_file_path}")
