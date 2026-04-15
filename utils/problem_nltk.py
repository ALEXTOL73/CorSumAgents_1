import nltk
custom_nltk_path = ""
nltk.data.path.append('/Users/AlexToledov/nltk_data')
nltk.download('stopwords', download_dir=custom_nltk_path)
nltk.download('punkt', download_dir=custom_nltk_path)
nltk.download('wordnet', download_dir=custom_nltk_path)
nltk.download('omw-1.4', download_dir=custom_nltk_path)
nltk.download('punkt_tab',download_dir=custom_nltk_path)
print("NLTK Data Paths:", nltk.data.path)