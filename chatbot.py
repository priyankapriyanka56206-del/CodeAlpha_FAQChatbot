"""
CodeAlpha Internship - Task 2: Chatbot for FAQs
NLP preprocessing (NLTK) + TF-IDF + cosine similarity.
Run:  python chatbot.py
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1. FAQ data (question, answer, alternative phrasings) ----------
FAQS = [
    ('How can I place an order?', 'Browse products, add items to your cart, click Checkout, enter your address and payment details, then confirm the order.', []),
    ('What payment methods do you accept?', 'We accept credit/debit cards, UPI, net banking, popular wallets, and Cash on Delivery for eligible orders.', []),
    ('How long does delivery take?', 'Standard delivery takes 3-7 business days. Express delivery takes 1-2 business days in major cities.', ['when will my order arrive', 'shipping time', 'delivery time']),
    ('How can I track my order?', 'Go to My Orders in your account and click Track Order, or use the tracking link sent to your email and SMS.', ['where is my package', 'track parcel', 'order status']),
    ('What is your return policy?', 'You can return most items within 7 days of delivery if they are unused and in original packaging.', ['money back policy', 'can I get my money back', 'send back item']),
    ('How do I return a product?', 'Open My Orders, select the item, click Return, choose a reason and schedule a pickup. We will arrange collection.', []),
    ('How long does a refund take?', 'Refunds are processed within 5-7 business days after we receive the returned item, to your original payment method.', ['when will I get my money back', 'refund time']),
    ('Can I cancel my order?', 'Yes. You can cancel from My Orders before the item is shipped. After shipping, you can refuse delivery or return it.', ['cancel purchase', 'stop my order']),
    ('How do I change my delivery address?', 'You can edit the address from My Orders before the order is shipped. After that, contact customer support.', []),
    ('Do you offer free shipping?', 'Yes, orders above Rs. 499 qualify for free standard shipping.', []),
    ('How do I reset my password?', 'Click Forgot Password on the login page, enter your registered email, and follow the reset link we send you.', []),
    ('How do I create an account?', 'Click Sign Up, enter your name, email and phone number, set a password, and verify your email.', []),
    ('Is my payment information secure?', 'Yes. All payments are encrypted and processed through PCI-DSS compliant payment gateways. We never store your full card details.', []),
    ('How can I contact customer support?', 'Email support@shopeasy.example or call 1800-000-000 between 9 AM and 9 PM, Monday to Saturday. You can also chat with us here.', []),
    ('What if I receive a damaged or wrong item?', 'Please report it within 48 hours of delivery from My Orders with a photo. We will replace it or refund you at no extra cost.', ['item arrived broken', 'defective product', 'received wrong product']),
    ('Do you ship internationally?', 'Currently we ship only within India. International shipping is planned for the future.', ['send abroad', 'deliver to other countries', 'overseas shipping', 'outside India']),
    ('How do I apply a discount coupon?', 'Enter the coupon code in the Apply Coupon box on the cart or checkout page and click Apply.', []),
    ('Do products come with a warranty?', "Electronics carry the manufacturer's warranty, mentioned on the product page. Keep your invoice for warranty claims.", []),
    ('Can I exchange a product for a different size or color?', 'Yes, eligible items can be exchanged within 7 days of delivery, subject to stock availability.', []),
    ('What are your working hours?', 'Our website is open 24/7. Customer support is available 9 AM to 9 PM, Monday to Saturday.', []),
]

# ---------- 2. NLP preprocessing ----------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(pkg, quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    word_tokenize("test")
    NLTK_OK = True
except Exception:  # fallback if NLTK data is unavailable
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP_WORDS
    NLTK_OK = False

STOP_WORDS = set(STOP_WORDS) - {"how", "what", "when", "where", "why", "can", "do"}


def preprocess(text):
    """lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize"""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = word_tokenize(text) if NLTK_OK else text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    if NLTK_OK:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------- 3. Matching with cosine similarity ----------
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_matrix = vectorizer.fit_transform(
    [preprocess(q + " " + " ".join(alt)) for q, a, alt in FAQS]
)


def get_response(user_input, threshold=0.20):
    msg = user_input.strip().lower()
    if msg in {"hi", "hello", "hey"}:
        return "Hello! Ask me about orders, delivery, returns or payments."
    scores = cosine_similarity(vectorizer.transform([preprocess(user_input)]), faq_matrix)[0]
    best = int(scores.argmax())
    if scores[best] < threshold:
        return "Sorry, I couldn't find an answer. Please rephrase or contact support@shopeasy.example."
    return FAQS[best][1]


# ---------- 4. Chat loop ----------
if __name__ == "__main__":
    print("ShopEasy FAQ Chatbot (type 'exit' to quit)")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye!")
            break
        if user:
            print("Bot:", get_response(user))
