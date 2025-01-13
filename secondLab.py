import os
import warnings
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Импортируем библиотеку для прогресс-бара

# Настройка логирования для подавления предупреждений от transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Отключение всех предупреждений
warnings.filterwarnings("ignore")

# Отключение предупреждения о symlinks на Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("[INFO] Загружаем очищенные данные...")
# Загрузка очищенных данных
cleaned_file = 'final_cleaned_youtube_data.csv'
data = pd.read_csv(cleaned_file)

# Проверка наличия необходимых столбцов
required_columns = ["cleaned_title", "category"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Файл должен содержать колонки: {required_columns}")
print("[INFO] Данные успешно загружены и проверены.")

# Преобразование категорий в числовые метки
print("[INFO] Преобразование категорий в числовые метки...")
label_encoder = LabelEncoder()
data["category"] = label_encoder.fit_transform(data["category"])
labels = data["category"].tolist()

# Настройка модели Robert
print("[INFO] Загружаем модель Roberta и токенизатор...")
model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(data["category"].unique()), ignore_mismatched_sizes=True
)
model.to(device)
print("[INFO] Модель успешно загружена.")

# Токенизация текстов
print("[INFO] Выполняется токенизация текстов...")
texts = data["cleaned_title"].astype(str).tolist()
tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
tokens = {key: value.to(device) for key, value in tokens.items()}
print("[INFO] Токенизация завершена.")

# Обучение модели и получение предсказаний
print("[INFO] Начинается процесс предсказания...")
batch_size = 16
predictions = []

# Используем tqdm для отображения прогресс-бара
with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Обработка батчей"):
        batch_tokens = {key: value[i:i + batch_size] for key, value in tokens.items()}
        outputs = model(**batch_tokens)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        predictions.extend(batch_predictions)

print("[INFO] Предсказания завершены.")

# Оценка качества модели
print("[INFO] Оцениваем качество модели...")
precision = precision_score(labels, predictions, average="weighted", zero_division=1)
recall = recall_score(labels, predictions, average="weighted")
f1 = f1_score(labels, predictions, average="weighted")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
