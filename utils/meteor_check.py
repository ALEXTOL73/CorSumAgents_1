import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

# Проверка токенизации
tokens = word_tokenize('Привет, как дела?')
print(f'✅ Токенизация: {tokens}')

# Проверка METEOR
ref = ['привет', 'как', 'дела']
hyp = ['привет', 'как', 'ты']
meteor = meteor_score([ref], hyp)
print(f'✅ METEOR: {meteor}')

print('✅ Все NLTK ресурсы работают корректно!')