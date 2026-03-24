import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# stopwords are common words like "the", "is", "and" that add no meaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Why: if a column is empty/missing, return empty string instead of crashing
    if not isinstance(text, str):
        return ""
    
    text = text.lower()                        # Why: "Job" and "job" are same word
    text = re.sub(r'http\S+', '', text)        # Why: URLs add no signal
    text = re.sub(r'[^a-z\s]', '', text)       # Why: remove numbers, punctuation
    text = re.sub(r'\s+', ' ', text).strip()   # Why: remove extra spaces
    
    # Why: remove stopwords so model focuses on meaningful words only
    words = [w for w in text.split() if w not in stop_words]
    
    return ' '.join(words)

def combine_and_clean(df):
    # Why: fill missing values with empty string so we don't get errors
    df = df.fillna('')
    
    # Why: combine all text columns — more text gives model more signal to learn from
    df['combined_text'] = (
        df['title'] + ' ' +
        df['company_profile'] + ' ' +
        df['description'] + ' ' +
        df['requirements'] + ' ' +
        df['benefits']
    )
    
    # Why: apply our clean_text function to every row
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    return df