from application.translator import Translator

# Training data
train_data = [
    ("你好", "hello"),
    ("早上好", "good morning"),
    ("再见", "goodbye"),
    ("谢谢", "thank you"),
    ("不客气", "you're welcome"),
    ("你叫什么名字", "what's your name"),
    ("今天天气如何", "how's the weather today"),
    ("我喜欢编程", "I love programming"),
    ("这是一个测试", "this is a test"),
    ("人工智能", "artificial intelligence"),
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
print("你好 →", translator.translate("你好"))
print("早上好 →", translator.translate("早上好"))
print("这是一个测试 →", translator.translate("这是一个测试"))
