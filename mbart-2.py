from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer

# Инициализация модели при старте сервера
class MBartSummarizer:
    def __init__(self, model_name="IlyaGusev/mbart_ru_sum_gazeta"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer.src_lang = "ru_RU"
        self.tokenizer.tgt_lang = "ru_RU"

    def summarize(self, text: str, max_length=150, min_length=40) -> str:
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

# Создание экземпляра модели
summarizer = MBartSummarizer()

# Инициализация FastAPI
app = FastAPI()

# Добавляем поддержку CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает запросы с любых доменов. Укажите точные домены для безопасности.
    allow_credentials=True,
    allow_methods=["*"],  # Разрешает все методы HTTP (GET, POST, OPTIONS и т.д.).
    allow_headers=["*"],  # Разрешает все заголовки.
)

# Описание запроса
class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(request: TextRequest):
    result = summarizer.summarize(request.text)
    return {"summary": result}

# Корневой маршрут
@app.get("/")
async def root():
    return {"message": "API для пересказа текста. Используйте /summarize для пересказа текста."}

# Обработчик для favicon
@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon provided"}

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
