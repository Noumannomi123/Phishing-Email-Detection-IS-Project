import streamlit as st
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
import string
import pickle
# Define a dictionary of chat word mappings
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}


stemmer = SnowballStemmer("english")
def extreme_clean(msg):
    new_msg = ''
    for word in msg.split():
        new_msg = new_msg + ' ' + stemmer.stem(word)
    msg = new_msg
    nopunc = [char for char in msg if char.lower() not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def replace_chat_words(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in chat_words:
            words[i] = chat_words[word.lower()]
    return ' '.join(words)
stop = stopwords.words('english')
def preprocess(msg: str):
    extreme_clean(msg)
    replace_chat_words(msg)
    msg = re.sub(r'[^a-zA-Z\s]', '', msg)
    msg = ' '.join([word for word in msg.split() if word not in (stop)])
    msg = msg.lower()
    msg = re.sub(r'\d+', '', msg)
    msg = re.sub(r'\s+', ' ', msg)
    msg = re.sub(r'[^\w\s]', '', msg)
    msg = re.sub(r'http\S+', '', msg)
    return msg 



with open("svm.pkl", "rb") as f:
    svm = pickle.load(f)
with open("knn.pkl", "rb") as f:
    knn = pickle.load(f)
with open("rfc.pkl", "rb") as f:
    rfc = pickle.load(f)
with open("nb.pkl", "rb") as f:
    nb = pickle.load(f)
with open('tf-idf.pkl', 'rb') as f:
    tf_idf = pickle.load(f)
with open('bow.pkl', 'rb') as f:
    bow = pickle.load(f)
def get_spam_ham(msg):
    return msg[0]


def sentence_builder(msg, model):
    msg = preprocess(msg)
    print(msg)
    msg = tf_idf.transform(bow.transform([msg]))
    if model == "SVM":
        return get_spam_ham(svm.predict(msg))
    elif model == "KNN":
        return get_spam_ham(knn.predict(msg))
    elif model == "RFC":
        return get_spam_ham(rfc.predict(msg))
    elif model == "MultinomialNB":
        return get_spam_ham(nb.predict(msg))

# Streamlit GUI
st.title("Sam vs Ham Detector")

# Input for both Email Text and URL in one field
input_text = st.text_area("Enter Message Text: ")

if st.button("Classify as Spam or Ham"):
    if input_text:  # If there is any input
        models = ['SVM', 'KNN', "RFC", 'MultinomialNB']
        model = 'SVM'
        # Prepare a list to hold results for each model
        results = []

        for model in models:
            result = sentence_builder(msg=input_text, model=model)
            
            if result:
                # Store the result for each model
                results.append([model, result])

        # Display the results as a table
        if results:
            st.subheader("Prediction Results")
            st.table(results)
        else:
            st.write("No results to display.")
            
    else:
        st.write("Please enter some text or a URL for prediction.")