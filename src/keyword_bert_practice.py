from transformers import DistilBertModel, DistilBertTokenizer
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# 입력 텍스트
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)

# [CLS] 토큰 임베딩 추출
embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

# 2. TF-IDF로 키워드 추출
corpus = [text]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 키워드 추출
feature_names = vectorizer.get_feature_names_out()
scores = X.toarray()[0]
keywords = [(feature_names[i], scores[i]) for i in range(len(scores))]
keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

print("키워드:", [word for word, score in keywords[:5]])