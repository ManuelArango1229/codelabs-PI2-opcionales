import re, random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import joblib

random.seed(42); np.random.seed(42)
spam = [
    "Gana dinero fácil en 24 horas, haz clic aquí",
    "Has sido seleccionado para un premio, comparte tus datos",
    "Tu cuenta será cerrada, verifica en este enlace",
    "Crypto inversión garantizada 10% diario",
    "Último aviso: paga ahora para evitar bloqueo de cuenta",
    "Te transfiero 1000 USD si completas este formulario",
    "Promoción exclusiva solo hoy, ingresa tu tarjeta",
    "Recarga gratis, solo confirma tu contraseña",
    "WhatsApp Premium sin costo, descarga aquí",
    "Factura vencida, ingresa a este link para pagar"
]

legit = [
    "Hola, necesito soporte para iniciar sesión",
    "¿Cuánto cuesta el plan anual y medios de pago?",
    "El pedido llegó tarde, ¿pueden ayudarme?",
    "Estoy interesado en una demo del producto",
    "¿Tienen descuento por volumen para empresas?",
    "Se cierra la app al abrir el carrito, por favor apoyo",
    "El envío llegó bien, gracias por la atención",
    "Deseo actualizar mi método de pago",
    "¿Cuál es el tiempo de entrega estimado?",
    "Quiero cambiar la contraseña de mi cuenta"
]

def variar(s):
    extras = ["", "!", "!!", " urgente", " por favor", " ahora", " hoy", " gratis"]
    return s + random.choice(extras)

data = []
for _ in range(25):
    data += [(variar(x), 1) for x in spam]
    data += [(variar(x), 0) for x in legit]

df = pd.DataFrame(data, columns=["texto","etiqueta"]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total muestras disponibles: {len(df)}  |  Spam: {df['etiqueta'].sum()}  |  Legítimos: {(1-df['etiqueta']).sum()}")

def limpiar(s):
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["texto_clean"] = df["texto"].apply(limpiar)

X_train, X_test, y_train, y_test = train_test_split(
    df["texto_clean"], df["etiqueta"],
    test_size=0.2, random_state=42, stratify=df["etiqueta"]
)

pipe = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1, solver="liblinear")
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\n--- MÉTRICAS INICIALES (umbral predeterminado 0.5) ---")
print(f"Exactitud en test: {acc:.3f}")

print("\nDesglose del desempeño por categoría:")
print(classification_report(y_test, pred, digits=3))

cm_base = pd.DataFrame(
    confusion_matrix(y_test, pred, labels=[0,1]),
    index=["Etiqueta_real:Legit", "Etiqueta_real:Spam"],
    columns=["Predicción:Legit", "Predicción:Spam"]
)

print("\nTabla de confusión (base):\n")
print(cm_base)

probs = pipe.predict_proba(X_test)[:, 1]
ap = average_precision_score(y_test, probs)
precision, recall, thresholds = precision_recall_curve(y_test, probs)

print(f"\nÁrea bajo curva Precisión-Recall (AP Score): {ap:.3f}")

umbral = 0.7
pred_u = (probs >= umbral).astype(int)
acc_u = accuracy_score(y_test, pred_u)

print(f"\n--- MÉTRICAS CON UMBRAL AJUSTADO ({umbral}) ---")
print(f"Exactitud ajustada: {acc_u:.3f}")

print("\nInforme detallado (umbral ajustado):")
print(classification_report(y_test, pred_u, digits=3))

cm_adj = pd.DataFrame(
    confusion_matrix(y_test, pred_u, labels=[0,1]),
    index=["Real_Legit", "Real_Spam"],
    columns=["Pred_Legit", "Pred_Spam"]
)

print("\nTabla de confusión ajustada:\n")
print(cm_adj)

plt.figure()
plt.step(recall, precision, where="post")
plt.xlabel("Sensibilidad (Recall)")
plt.ylabel("Precisión")
plt.title("Precisión–Recall: guía para selección de umbral")
plt.show()

scores = cross_val_score(pipe, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")
print("\n--- VALIDACIÓN CRUZADA (5-fold) ---")
print(f"F1_macro promedio: {scores.mean():.3f}")
print(f"Variabilidad (STD): {scores.std():.3f}")

def clasificar_mensajes(textos, threshold=0.7):
    tx = [limpiar(t) for t in textos]
    prob = pipe.predict_proba(tx)[:, 1]
    yhat = (prob >= threshold).astype(int)
    etiquetas = ["spam/estafa" if x==1 else "legítimo" for x in yhat]
    return list(zip(textos, prob.round(3), etiquetas))

nuevos = [
    "Has sido seleccionado para un premio, comparte tus datos aquí",
    "Necesito recuperar acceso a mi cuenta, me pueden ayudar?",
    "Gana dinero rápido hoy, sin riesgo, solo ingresa tu tarjeta",
    "Hola, ¿cuál es el precio del plan anual y si aceptan tarjeta?"
]

print("\n--- CLASIFICACIÓN DE MENSAJES RECIENTES (umbral 0.7) ---")
for texto, prob, clase in clasificar_mensajes(nuevos, threshold=0.7):
    print(f"Mensaje: '{texto}'")
    print(f"  → Probabilidad estimada de spam: {prob}")
    print(f"  → Clasificación asignada: {clase}\n")

joblib.dump(pipe, "pipeline_spam.joblib")
print("\nModelo almacenado en archivo: pipeline_spam.joblib")

loaded = joblib.load("pipeline_spam.joblib")
print("Verificación tras carga:", loaded.predict(["Gana dinero fácil completando este formulario"])[0])
