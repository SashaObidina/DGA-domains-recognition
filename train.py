import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import re
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy.stats import entropy


def split_domain(domain):
    dot_index = domain.rfind('.') # поиск последней позиции точки
    if dot_index != -1:
        before_dot = domain[:dot_index]
        after_dot = domain[dot_index+1:] if dot_index + 1 < len(domain) else ''
    else:
        before_dot = domain
        after_dot = ''
    return before_dot, after_dot
    

def calculate_features(domain):
    # Длина доменного имени второго уровня
    length = len(domain)

    # Количество тире в доменном имени
    num_hyphens = domain.count('-')

    # Количество цифр в доменном имени
    num_digits = sum(c.isdigit() for c in domain)

    # Количество полных чисел в доменном имени
    full_numbers = re.findall(r'\d+', domain)
    num_full_numbers = len(full_numbers)

    # Подсчет частоты каждого символа в доменном имени
    sym_counts = Counter(domain)
    total_symbols = sum(sym_counts.values())
    sym_freqs = {sym: count / total_symbols for sym, count in sym_counts.items()}

    # Вычисление энтропии на основе распределения частот символов
    ent = entropy(list(sym_counts.values()), base=2)

    return [length, num_hyphens, num_digits, num_full_numbers, ent]
    

# Определение модели
def create_model():
    domain = tf.keras.Input(shape = (maxlen,), name='domain')
    add_features = tf.keras.Input(shape = (add_features_shape,), name='add_features')

    domen_emb = layers.Embedding(input_dim=max_features_num, output_dim=embed_dim, input_length=maxlen)(domain)
    bilstm = layers.Bidirectional(layers.LSTM(hidden_dim_bilstm))(domen_emb)

    conv1 = layers.Conv1D(conv_hidden_dims[1], kernel_sizes[0], activation='relu')(domen_emb)
    conv2 = layers.Conv1D(conv_hidden_dims[1], kernel_sizes[1], activation='relu')(domen_emb)
    pool1 = layers.GlobalMaxPooling1D()(conv1)
    pool2 = layers.GlobalMaxPooling1D()(conv2)
    output1 = layers.add([pool1, pool2, bilstm])

    dense1 = layers.Dense(16, activation='relu')(add_features)
    dropout1 = layers.Dropout(dropout_prob)(dense1)
    output2 = layers.concatenate([output1, dense1])
    flattened = layers.Flatten()(output2)
    dropout2 = layers.Dropout(dropout_prob)(flattened)
    dense2 = layers.Dense(128, activation='relu')(dropout2)
    dense3 = layers.Dense(64, activation='relu')(dense2)
    output = layers.Dense(1, activation='sigmoid')(dense3)

    model = models.Model(inputs=[domain, add_features], outputs=output)
    return model


if __name__ == '__main__':
    embed_dim = 128
    embed_dim_tld = 64
    hidden_dim_bilstm = 128
    kernel_sizes = [4, 2]  # Размеры ядер свертки
    conv_hidden_dims = [128, 256]
    lstm_hidden_dim = 32
    dropout_prob = 0.5
    add_features_shape = 5
    
    df = pd.read_csv('/dga_domains.csv', header=None)
    df['is_dga'] = df[0].apply(lambda x: 1 if x == 'dga' else 0)
    result_df = df[[2, 'is_dga']]
    result_df.columns = ['domain', 'is_dga']

    domains = result_df['domain'].values
    labels = result_df['is_dga'].values
    
    # Создается словарь уникальных символов, домены переводятся в числовые строки с паддингами-нулями - в эмбеддинги
    unique_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.'
    chars_dict = {x: i + 1 for i, x in enumerate(unique_chars)}  # Индексация начинается с 1, 0 оставляем для паддинга
    max_features_num = len(chars_dict) + 1

    # Разделение доменов второго и первого уровней
    domains_ = []
    tlds_ = []

    for domain in domains:
        second_level, first_level = split_domain(domain)
        domains_.append(second_level)
        tlds_.append(first_level)

    # Дополнительные признаки для домена 2-го уровня - длина, количество тире, кол-во цифр, кол-во чисел, энтропия
    add_features = [calculate_features(domain) for domain in domains_]

    maxlen = np.max([len(x) for x in domains])
    maxlen_tld = np.max([len(x) for x in tlds_])
    domains_int = [[chars_dict[c] for c in domain] for domain in domains]
    domains_padded = [domain + [0] * (maxlen - len(domain)) for domain in domains_int]
    tlds_int = [[chars_dict[c] for c in tld] for tld in tlds_]
    tlds_padded = [tld + [0] * (maxlen_tld - len(tld)) for tld in tlds_int]

    # Данные делятся на обучающую+валидационную и тестовую выборки
    X_train_domains, X_test_domains, X_train_tlds, X_test_tlds, y_train, y_test = train_test_split(domains_padded, tlds_padded, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train_add, X_test_add = train_test_split(add_features, test_size=0.2, random_state=42, stratify=labels)

    X_train_domains = np.array(X_train_domains)
    X_test_domains = np.array(X_test_domains)

    X_train_tlds = np.array(X_train_tlds)
    X_test_tlds = np.array(X_test_tlds)

    X_train_add = np.array(X_train_add)
    X_test_add = np.array(X_test_add)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = create_model()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, to_file='/model.png', show_shapes=True)
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit([X_train_domains, X_train_add], y_train, batch_size=128, epochs=50, validation_split=0.1, callbacks=[earlyStopping])
    model.save('/my_model8.h5')
