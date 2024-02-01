import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    preprocessed_sentences = []
    stop_words = set(stopwords.words("english"))
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(" ".join(filtered_words))
        
    return preprocessed_sentences

def calculate_coherence(text):
    preprocessed_sentences = preprocess_text(text)
    
    if len(preprocessed_sentences) < 2:
        return 0  # Return 0 coherence if there are less than 2 sentences
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_sentences)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[1:])
    average_similarity = cosine_similarities.mean()
    
    return average_similarity

if __name__ == "__main__":
    input_text = input("Enter the text to evaluate coherence: ")
    coherence_score = calculate_coherence(input_text)
    print("Coherence Score:", coherence_score)
