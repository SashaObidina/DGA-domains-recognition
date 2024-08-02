from train import *
from tensorflow.keras.models import load_model

model = load_model('/my_model8.h5')

test_data = pd.read_csv('/test.csv')
test_domains = test_data['domain'].values

# Разделение доменов второго и первого уровней
test_domains_ = []
test_tlds_ = []

for domain in test_domains:
    second_level, first_level = split_domain(domain)
    test_domains_.append(second_level)
    test_tlds_.append(first_level)
    
# Дополнительные признаки для домена 2-го уровня - длина, количество тире, кол-во цифр, кол-во чисел, энтропия
add_features_val = [calculate_features(domain) for domain in test_domains_]

test_domains_int = [[chars_dict[c] for c in domain] for domain in test_domains]
test_domains_padded = [domain + [0] * (maxlen - len(domain)) for domain in test_domains_int]
test_tlds_int = [[chars_dict[c] for c in tld] for tld in test_tlds_]
test_tlds_padded = [tld + [0] * (maxlen_tld - len(tld)) for tld in test_tlds_int]

y_test_pred_task = model.predict([np.array(test_domains_padded), np.array(add_features_val)])
y_test_pred_task = (y_test_pred_task > 0.5).astype(int)

predictions_df = pd.DataFrame({
    'domain': test_domains,
    'is_dga': y_test_pred_task.flatten()
})

predictions_df.to_csv('/prediction.csv', index=False)