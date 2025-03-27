from application.translator import Translator

# Training data
train_data = [
    ("bonjour", "hello"),
    ("bon matin", "good morning"),
    ("au revoir", "goodbye"),
    ("merci", "thank you"),
    ("de rien", "you're welcome"),
    ("comment vous appelez-vous", "what's your name"),
    ("quel temps fait-il aujourd'hui", "how's the weather today"),
    ("j'aime programmer", "I love programming"),
    ("c'est un test", "this is a test"),
    ("intelligence artificielle", "artificial intelligence"),
]

# Initialize translator
translator = Translator(
    train_data=train_data,
    d_model=128,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    batch_size=32,
    epochs=10,
    dropout=0.1,
    max_seq_len=10,
)

# Training
translator.train()

# Test translations
print("\nTranslation examples:")
print("bonjour →", translator.translate("bonjour"))
print("bon matin →", translator.translate("bon matin"))
print("c'est un test →", translator.translate("c'est un test"))
