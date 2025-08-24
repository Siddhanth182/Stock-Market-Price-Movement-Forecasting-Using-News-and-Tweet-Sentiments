import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xgboost import XGBClassifier
import pytz
from tqdm import tqdm

# Load your data
# tsla_stock_df = pd.read_csv("nvda-2025-01-01___2025-04-30.csv", parse_dates=['Datetime'])
# news_tsla_df_orig = pd.read_csv("Iter6Nvidia_news_filtered_nyse_hours.csv")

tsla_stock_df = pd.read_csv("TSLA-2025-01-01___2025-04-18.csv", parse_dates=['Datetime'])
news_tsla_df_orig = pd.read_csv("Copy of Iter2tesla_news_filtered_nyse_hours.csv")

# tsla_stock_df = pd.read_csv("AMZNnew_filtered_stock_data.csv", parse_dates=['Datetime'])
# news_tsla_df_orig = pd.read_csv("Iter5AMAZON_news_filtered_nyse_hours.csv")
nasdaq_df = pd.read_csv("nsdq-2025-01-01___2025-04-30.csv", parse_dates=['Datetime'])

nasdaq_df['Datetime'] = pd.to_datetime(nasdaq_df['Datetime'], utc=True)
nasdaq_df = nasdaq_df.sort_values('Datetime')
nasdaq_df['nasdaq_price_change_pct'] = nasdaq_df['Close'].pct_change() * 100

news_tsla_df_orig['publishedAt_ny'] = pd.to_datetime(news_tsla_df_orig['publishedAt_ny'])
tsla_stock_df['Datetime'] = pd.to_datetime(tsla_stock_df['Datetime'], utc=True)

news_dates = news_tsla_df_orig['publishedAt_ny'].dt.date.unique()
stock_dates = tsla_stock_df['Datetime'].dt.tz_convert('US/Eastern').dt.date.unique()
valid_dates = set(news_dates) & set(stock_dates)
news_tsla_f = news_tsla_df_orig[news_tsla_df_orig['publishedAt_ny'].dt.date.isin(valid_dates)].copy()

ny_tz = pytz.timezone("US/Eastern")
news_df = news_tsla_f.copy()
stock_df = tsla_stock_df.copy()
stock_df['Datetime'] = stock_df['Datetime'].dt.tz_convert(ny_tz).dt.tz_localize(None)
news_df['publishedAt_ny'] = news_df['publishedAt_ny'].dt.tz_convert(ny_tz).dt.tz_localize(None)
news_df['hour_bin'] = pd.to_datetime(news_df['hour_bin'], utc=True).dt.tz_convert(ny_tz).dt.tz_localize(None)

news_df = news_df.sort_values('hour_bin')
stock_df = stock_df.sort_values('Datetime')

merged_df = pd.merge_asof(
    news_df,
    stock_df[['Datetime', 'Close']],
    left_on='hour_bin',
    right_on='Datetime',
    direction='backward'
).rename(columns={'Close': 'price_at_news', 'Datetime': 'matched_stock_time'})

price_lookup = stock_df.set_index('Datetime')['Close']
merged_df['next_hour'] = merged_df['matched_stock_time'] + pd.Timedelta(hours=1)
merged_df['price_after_news'] = merged_df['next_hour'].map(price_lookup)
merged_df['price_change_pct_next_hour'] = (merged_df['price_after_news'] - merged_df['price_at_news']) / merged_df['price_at_news'] * 100
merged_df['price_moved_up'] = (merged_df['price_change_pct_next_hour'] > 0.5).astype(int)

# FinBERT inference
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
tqdm.pandas()
label_map = model.config.id2label

def sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
    return {f"{label_map[i].lower()}_score": probs[i].item() for i in range(len(probs))}

sentiment_df = merged_df['title'].progress_apply(sentiment_score).apply(pd.Series)
merged_df = pd.concat([merged_df, sentiment_df], axis=1)
merged_df['sentiment_newmodel'] = sentiment_df.idxmax(axis=1).str.replace('_score', '')
merged_df = merged_df[merged_df['sentiment_newmodel'] != 'neutral'].copy()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_roc(series, window=12):
    return (series.diff(window) / series.shift(window)) * 100

stock_df['rsi'] = compute_rsi(stock_df['Close'])
stock_df['momentum'] = compute_roc(stock_df['Close'])
stock_df['price_change_pct'] = stock_df['Close'].pct_change() * 100

merged_df = pd.merge(
    merged_df,
    stock_df[['Datetime', 'Volume', 'rsi', 'momentum', 'price_change_pct']],
    left_on='matched_stock_time',
    right_on='Datetime',
    how='left'
)

merged_df['volume_z'] = (merged_df['Volume'] - merged_df['Volume'].rolling(10).mean()) / merged_df['Volume'].rolling(10).std()
merged_df['volume_spike'] = merged_df['volume_z'] > 1

agg_sentiment = merged_df.groupby('hour_bin').agg({
    'positive_score': 'mean',
    'negative_score': 'mean'
}).reset_index()
agg_sentiment['avg_sentiment_score'] = agg_sentiment['positive_score'] - agg_sentiment['negative_score']
final_df = merged_df.merge(agg_sentiment, on='hour_bin', suffixes=('', '_agg'))

final_df['matched_stock_time'] = pd.to_datetime(final_df['matched_stock_time'], utc=True)
final_df = pd.merge_asof(
    final_df.sort_values('matched_stock_time'),
    nasdaq_df[['Datetime', 'Close', 'nasdaq_price_change_pct']].rename(columns={'Close': 'nasdaq_close'}),
    left_on='matched_stock_time',
    right_on='Datetime',
    direction='backward'
)

final_df['rolling_sentiment_3h'] = final_df['avg_sentiment_score'].rolling(window=3, min_periods=1).mean()
final_df['rolling_price_change_3h'] = final_df['price_change_pct'].rolling(window=3, min_periods=1).sum()

final_df.dropna(subset=[
    'positive_score', 'negative_score', 'avg_sentiment_score',
    'rsi', 'momentum', 'volume_spike', 'rolling_sentiment_3h',
    'rolling_price_change_3h', 'nasdaq_price_change_pct'
], inplace=True)


# 1. Prepare features and label
features = [
    'positive_score', 'negative_score', 'avg_sentiment_score', 
    'rsi', 'momentum', 'volume_spike', 
    'rolling_sentiment_3h', 'rolling_price_change_3h',
    'nasdaq_price_change_pct'
]

X = final_df[features].copy()
X['volume_spike'] = X['volume_spike'].astype(int)
y = final_df['price_moved_up']

# 2. Split consistently
split_index = int(0.8 * len(X))
X_train = X.iloc[:split_index].copy()
y_train = y.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()
y_test = y.iloc[split_index:].copy()

train_df = final_df.iloc[:split_index].copy()
test_df = final_df.iloc[split_index:].copy()

# 3. Train model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 4. Add predictions
train_df['predicted_up'] = xgb_model.predict(X_train)
train_df['prob_up'] = xgb_model.predict_proba(X_train)[:, 1]
test_df['predicted_up'] = xgb_model.predict(X_test)
test_df['prob_up'] = xgb_model.predict_proba(X_test)[:, 1]

# 5. Save everything
xgb_model.save_model("xgb_nvda_model.json")
# train_df.to_csv("nvda_train_predictions.csv", index=False)
# test_df.to_csv("nvda_test_predictions.csv", index=False)
train_df.to_csv("tsla_train_predictions.csv", index=False)
test_df.to_csv("tsla_test_predictions.csv", index=False)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluate on test data
print("=== Accuracy Report on Test Data ===")
y_pred_test = test_df['predicted_up']
y_true_test = y_test

print("Confusion Matrix:")
print(confusion_matrix(y_true_test, y_pred_test))
print("\nClassification Report:")
print(classification_report(y_true_test, y_pred_test))
print(f"Accuracy: {accuracy_score(y_true_test, y_pred_test):.4f}")