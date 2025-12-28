import pandas as pd
import numpy as np
import os
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

def recall_at_k(model, user_items_train, user_items_test, k=10, num_threads=1):
    n_users = user_items_test.shape[0]
    total_recall = 0.0
    n_evaluated_users = 0

    for user_id in tqdm(range(n_users), desc="Оценка Recall_at_k", disable=(num_threads != 1)):
        if user_items_test[user_id].nnz == 0:
            continue

        recommended_items, _ = model.recommend(
            user_id,
            user_items_train[user_id],
            N=k,
            filter_already_liked_items=True
        )

        test_items = set(user_items_test[user_id].nonzero()[1])
        if not test_items:
            continue

        hits = len(set(recommended_items) & test_items)
        recall = hits / len(test_items)

        total_recall += recall
        n_evaluated_users += 1

    return total_recall / n_evaluated_users if n_evaluated_users > 0 else 0.0

def load_item_categories():
    files = ['./resources/item_properties_part1.csv', './resources/item_properties_part2.csv']
    dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
        else:
            print(f"Файл {f} не найден")
            return None

    props = pd.concat(dfs, ignore_index=True)
    cat_props = props[props['property'] == 'categoryid'].copy()
    cat_props['categoryid'] = pd.to_numeric(cat_props['value'], errors='coerce')
    cat_props = cat_props.dropna(subset=['categoryid'])
    cat_props['categoryid'] = cat_props['categoryid'].astype(int)

    item_to_category = cat_props[['itemid', 'categoryid']].drop_duplicates()
    print(f"Загружено {len(item_to_category)} товаров с категориями")
    return item_to_category

def visualize_item_embeddings(model, item_encoder, item_to_category, n_samples=3000):
    print("Визуализация эмбеддингов товаров")

    item_indices = np.arange(len(model.item_factors))
    original_itemids = item_encoder.inverse_transform(item_indices)

    df_items = pd.DataFrame({'itemid': original_itemids, 'internal_id': item_indices})
    df_items = df_items.merge(item_to_category, on='itemid', how='left')
    df_items = df_items.dropna(subset=['categoryid'])
    df_items['categoryid'] = df_items['categoryid'].astype(int)

    top_cats = df_items['categoryid'].value_counts().head(10).index
    df_items = df_items[df_items['categoryid'].isin(top_cats)]

    if len(df_items) > n_samples:
        df_items = df_items.sample(n=n_samples, random_state=42)
    if len(df_items) == 0:
        print("Нет данных для визуализации")
        return

    item_factors = model.item_factors[df_items['internal_id'].values]
    print("Запуск UMAP")
    embeddings_2d = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(item_factors)

    df_items['x'], df_items['y'] = embeddings_2d[:, 0], embeddings_2d[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_items, x='x', y='y', hue='categoryid', palette='tab10', alpha=0.7, s=25)
    plt.title("Эмбеддинги товаров", fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Категория", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("item_embeddings_umap.png", dpi=150)
    plt.show()
    print("График сохранён")

if __name__ == "__main__":
    # Загружаем данные из дата-сета
    print("Загружаем дата-сет")
    events = pd.read_csv('./resources/events.csv')

    event_weights = {'view': 1, 'addtocart': 3, 'transaction': 5}
    events = events[events['event'].isin(event_weights)].copy()
    events['weight'] = events['event'].map(event_weights)
    print(f"Загружено {len(events)} событий")

    # Фильтруем пользователей
    user_counts = events['visitorid'].value_counts()
    valid_visitors = user_counts[user_counts >= 2].index
    events = events[events['visitorid'].isin(valid_visitors)].copy()
    print(f"После фильтрации: {len(events)} событий")


    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    events['user_id'] = user_encoder.fit_transform(events['visitorid'])
    events['item_id'] = item_encoder.fit_transform(events['itemid'])

    n_users = events['user_id'].nunique()
    n_items = events['item_id'].nunique()
    print(f"Закодировано: {n_users} пользователей, {n_items} товаров")

    # Формируем обучающие и тестовые данные
    events = events.sort_values('timestamp')
    last_event_idx = events.groupby('user_id')['timestamp'].idxmax()
    test_events = events.loc[last_event_idx]
    train_events = events.drop(index=last_event_idx)
    print(f"Обучающих: {len(train_events)} событий, Тестовых: {len(test_events)} событий.")

    # Преобразуем в матрицы
    def build_matrix(df, n_u, n_i):
        return coo_matrix(
            (df['weight'].values, (df['user_id'].values, df['item_id'].values)),
            shape=(n_u, n_i)
        ).tocsr()

    train_matrix = build_matrix(train_events, n_users, n_items)
    test_matrix = build_matrix(test_events, n_users, n_items)

    # Обучаем модель
    print("Обучение модели")
    model = AlternatingLeastSquares(
        factors=32,
        regularization=0.01,
        iterations=15,
        random_state=42,
        use_gpu=False
    )
    model.fit(train_matrix)
    print("Модель обучена")

    # Проверка размеров
    print(f"user_factors: {model.user_factors.shape}")
    print(f"item_factors: {model.item_factors.shape}")

    # Проверяем Recal_at_k
    train_binary = train_matrix.copy()
    train_binary.data = np.ones_like(train_binary.data)

    test_binary = test_matrix.copy()
    test_binary.data = np.ones_like(test_binary.data)

    rec_at_10 = recall_at_k(model, train_binary, test_binary, k=10, num_threads=1)
    print(f"Recall_at_k: {rec_at_10:.4f}")

    # Формируем рекомендации
    def recommend(user_original_id, k=10):
        if user_original_id not in user_encoder.classes_:
            print(f"Пользователь {user_original_id} не найден")
            return []
        user_idx = user_encoder.transform([user_original_id])[0]
        item_ids, _ = model.recommend(
            user_idx,
            train_matrix[user_idx],
            N=k,
            filter_already_liked_items=True
        )
        return item_encoder.inverse_transform(item_ids).tolist()

    sample_user = train_events['visitorid'].iloc[1000]
    recs = recommend(sample_user, k=5)
    print(f"Пользователь {sample_user} - рекомендации: {recs}")

    # Визуализация
    print("Загрузка категорий")
    item_to_category = load_item_categories()
    if item_to_category is not None:
        visualize_item_embeddings(model, item_encoder, item_to_category)
    else:
        print("Нет данных для визуализации")