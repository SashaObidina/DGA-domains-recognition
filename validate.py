from train import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

model = load_model('/my_model8.h5')

val_data = pd.read_csv('/val.csv')
val_domains = val_data['domain'].values
val_labels = val_data['is_dga'].values

# Разделение доменов второго и первого уровней
val_domains_ = []
val_tlds_ = []

for domain in val_domains:
    second_level, first_level = split_domain(domain)
    val_domains_.append(second_level)
    val_tlds_.append(first_level)

# Дополнительные признаки для домена 2-го уровня - длина, количество тире, кол-во цифр, кол-во чисел, энтропия
add_features_val = [calculate_features(domain) for domain in val_domains_]

val_domains_int = [[chars_dict[c] for c in domain] for domain in val_domains]
val_domains_padded = [domain + [0] * (maxlen - len(domain)) for domain in val_domains_int]
val_tlds_int = [[chars_dict[c] for c in tld] for tld in val_tlds_]
val_tlds_padded = [tld + [0] * (maxlen_tld - len(tld)) for tld in val_tlds_int]

y_val_pred = model.predict([np.array(val_domains_padded), np.array(add_features_val)])

y_val_pred = (y_val_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(val_labels, y_val_pred)
true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()

accuracy = accuracy_score(val_labels, y_val_pred)
precision = precision_score(val_labels, y_val_pred)
recall = recall_score(val_labels, y_val_pred)
f1 = f1_score(val_labels, y_val_pred)

results = f"""True positive: {true_positive}
False positive: {false_positive}
False negative: {false_negative}
True negative: {true_negative}
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1: {f1:.4f}"""

with open('/validation.txt', 'w') as file:
    file.write(results)
