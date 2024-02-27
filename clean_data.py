import pandas as pd
import re
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize

# Function to remove sentences containing specific keywords or phrases from text
# def remove_sentences(text, keywords):
#     sentences = sent_tokenize(text)
#     cleaned_sentences = [sentence for sentence in sentences if not any(keyword in sentence.lower() for keyword in keywords)]
#     cleaned_text = ' '.join(cleaned_sentences)
#     return cleaned_text

# # Function to clean description text
# def clean_description(description):
#     if pd.isna(description):  # Check if description is NaN
#         return ''  # Return empty string if NaN
    
#     # Convert description to lowercase
#     description = description.lower()
    
#     # Define custom stopwords related to author information
#     author_related_stopwords = {'contact', 'shipping quote', 'local pickup', 'phone', 'email', 'address', 'location', 'return', 'exchange', 'visit', 'please see the photos', 'please see', 'shipping', 'dimension', 'call or text', 'we offer unique', 'please feel free to', 'safe delivery', 'blanket wrap', 'pickup', 'happy to help', 'elle', 'meet us', 'fort my', 'warehouse', 'interested', 'template not found', 'sold as', 'confirms age', 'unique character'}
    
#     # Define additional words to remove
#     additional_stopwords = {'new', 'package', 'description', 'interested','seeing' ,'brand', 'template', 'found', 'category', 'height', 'width' , 'depth', 'item', 'number', 'id', 'please', 'see', 'photo', 'photo', 'shipping', 'quote', 'local', 'pickup', 'phone', 'email', 'address', 'location', 'return', 'exchange', 'visit', 'please see the photos', 'please see', 'shipping', 'dimension', 'call or text', 'we offer unique', 'please feel free to', 'safe delivery', 'blanket wrap', 'pickup', 'happy to help', 'elle', 'meet us', 'fort my', 'warehouse', 'interested', 'template not found', 'condition', 'sku', 'tags', 'heigt', 'shipper', 'suggestion', 'shipping', 'quote', 'local', 'pickup', 'phone', 'email', 'address', 'location', 'return', 'exchange', 'visit', 'please see the photos', 'please see', 'shipping', 'dimension', 'call or text', 'we offer unique', 'please feel free to', 'safe delivery', 'blanket wrap', 'pickup', 'happy to help', 'elle', 'meet us', 'fort my', 'warehouse', 'interested', 'template not found', 'sold as' , 'description:', 'used', 'gently', 'blemishes', 'damage', 'included', 'photos', 'any', 'or', 'will', 'be', 'in', 'the', 'wide', 'x', 'deep', 'high', 'interior', 'measurements', 'are', 'outer', 'inner', 'center', 'stamped', 'all', 'open', 'close', 'smoothly', 'comes', 'from', 'smoke', 'free', 'estate', 'home', 'great', 'as', 'a', 'bureau', 'buffet', 'cabinet', 'sideboard', 'credenza', 'media', 'center', 'changing', 'table', 'nursery', 'am', 'and', 'my', 'husband', 'experts', 'at', 'sourcing', 'treasures', 'feel', 'ask', 'many', 'questions', 'request', 'pictures', 'youd', 'like', 'before', 'ordering', 'happy', 'help', 'upon', 'receiving', 'your', 'it', 'is', 'responsibility', 'fully', 'inspect', 'piece', 'before', 'furniture', 'leaves', 'contact', 'me', 'immediately', 'if', 'there', 'issue', 'we', 'offer', 'unique', 'vintage', 'item', 'may', 'have', 'imperfections', 'please', 'feel', 'free', 'ask', 'as', 'many', 'questions', 'request', 'as', 'many', 'pictures', 'as', 'you', 'like', 'before', 'ordering', 'i', 'am', 'happy', 'to', 'help', 'elle', 'ready', 'enjoyed'}
    
#     # Remove patterns matching specific format
#     description = re.sub(r'(\bheight\s*=\s*\d+\s*\'\'\s*h\s*\bdepth\s*=\s*\d+\s*\'\'\s*d\s*\bwidth\s*=\s*\d+\s*\'\'\s*w\s*\bitem\s*number\s*:\s*\d+-\d+\s*\bitem\s*id\s*:\s*\d+\b)', '', description)
    
#     # Remove sentences containing author-related keywords
#     description = remove_sentences(description, author_related_stopwords)
    
#     # Remove special characters and digits
#     description = re.sub(r'[^a-zA-Z\s]', '', description)

#     # Tokenize description
#     tokens = word_tokenize(description)
    
#     # Remove additional stopwords
#     tokens = [word for word in tokens if word not in additional_stopwords]
    
#     # Lemmatize words
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Join tokens back into string
#     cleaned_description = ' '.join(tokens)
    
#     return cleaned_description

# Function to clean title text
def clean_title(title):
    if pd.isna(title):  # Check if title is NaN
        return ''  # Return empty string if NaN
    
    # Convert title to lowercase
    # title = title.lower()
    
    # Remove special characters and digits
    # cleaned_title = re.sub(r'[^a-zA-Z\s]', '', title)
    
    return title
