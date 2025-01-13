import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
# Пользовательский список стоп-слов для удаления из текста
stop_words = {
    "subscribe", "subscribed", "subscribers", "credits", "find", "social", "media",
    "notifications", "notification", "clicking", "bell", "icon", "show", "more",
    "less", "instagram", "twitter", "facebook", "gmail", "contact", "channel",
    "isnt", "share", "email", "http", "www", "the", "a", "and", "of", "to", "in",
    "on", "for", "with", "this", "that", "by", "it", "at", "from", "or", "as", "about"
}

# Функция для удаления эмодзи из текста
def remove_emoji(text):
    emoji_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Функция для очистки текста
# 1. Приведение текста к нижнему регистру
# 2. Удаление ссылок, пунктуации, чисел и лишних пробелов
# 3. Токенизация текста и удаление стоп-слов
# 4. Лемматизация вместо стемминга

def clean_text_advanced(text):
    lemmatizer = WordNetLemmatizer()

    # Приведение текста к нижнему регистру
    text = text.lower()

    # Удаление URL-ссылок
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Удаление пунктуации, чисел и специальных символов
    text = re.sub(r"[^\w\s]", " ", text)  # Удаление пунктуации
    text = re.sub(r"\d+", "", text)       # Удаление чисел
    text = re.sub(r"\s+", " ", text).strip()  # Удаление лишних пробелов

    # Удаление эмодзи
    text = remove_emoji(text)

    # Токенизация и удаление стоп-слов
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Возврат очищенного текста
    return " ".join(cleaned_tokens)

# Чтение исходного файла и очистка данных
data = pd.read_csv("youtube.csv")
data['cleaned_title'] = data['title'].apply(clean_text_advanced)
data['cleaned_description'] = data['description'].apply(clean_text_advanced)

# Сохранение очищенного файла
data.to_csv("final_cleaned_youtube_data.csv", index=False)
print("Очищенные данные сохранены в final_cleaned_youtube_data.csv")

# Пример для тестирования работы функции
sample_text = "Subscribe to our channel for more amazing content! Click the bell \U0001F514 for notifications!"
cleaned_sample = clean_text_advanced(sample_text)
print(cleaned_sample)
