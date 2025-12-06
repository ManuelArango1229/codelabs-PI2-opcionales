import re, random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

random.seed(42); np.random.seed(42)

ventas = [
    "Quiero saber el precio del plan premium",
    "¿Tienen descuentos por volumen para empresas?",
    "¿Cómo puedo pagar? ¿Tarjeta o transferencia?",
    "Estoy interesado en comprar 10 unidades",
    "¿Cuánto cuesta el plan anual y cómo se factura?"
]
soporte = [
    "No puedo iniciar sesión, sale error 403",
    "La app se cierra al abrir el carrito",
    "La impresora no conecta por wifi, ya reinicié",
    "Se perdió mi pedido en la app, ayuda",
    "No me llega el código de verificación"
]
queja = [
    "El pedido llegó incompleto y nadie responde",
    "Muy mala atención, llegó tarde y mal empacado",
    "Estoy inconforme, el producto vino dañado",
    "Demasiada demora, pésimo servicio",
    "Me trataron mal por WhatsApp, muy groseros"
]

def variar(s):
    extras = ["", "!", "!!", " por favor", " urgente", " de verdad", " gracias"]
    return s + random.choice(extras)

data = []
for _ in range(20):
    data += [(variar(x), "ventas") for x in ventas]
    data += [(variar(x), "soporte") for x in soporte]
    data += [(variar(x), "queja")   for x in queja]

df = pd.DataFrame(data, columns=["texto","etiqueta"]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Total de registros:", len(df), "| Distribución:", dict(df["etiqueta"].value_counts()))

def limpiar(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["texto_clean"] = df["texto"].apply(limpiar)

X_train, X_test, y_train, y_test = train_test_split(
    df["texto_clean"], df["etiqueta"], test_size=0.2, random_state=42, stratify=df["etiqueta"]
)

pipe = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LinearSVC(class_weight="balanced", random_state=42)
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"\nRendimiento general (accuracy): {acc:.3f}\n")
print("Desglose por categoría:\n")
print(classification_report(y_test, pred, digits=3))

cm = confusion_matrix(y_test, pred, labels=["ventas","soporte","queja"])
cm_df = pd.DataFrame(cm,
    index=["ref_ventas","ref_soporte","ref_queja"],
    columns=["pred_ventas","pred_soporte","pred_queja"]
)
print("\nMatriz comparativa real vs predicción:\n")
print(cm_df)

scores = cross_val_score(pipe, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")
print(f"\nValidación cruzada (F1_macro): promedio={scores.mean():.3f} | variabilidad={scores.std():.3f}")

def enrutar_mensajes(textos):
    tx = [limpiar(t) for t in textos]
    etiquetas = pipe.predict(tx)
    area = {"ventas":"Área Comercial", "soporte":"Soporte Técnico", "queja":"Atención al Usuario"}
    rutas = [area[e] for e in etiquetas]
    return list(zip(textos, etiquetas, rutas))

nuevos = [
    "Se dañó el botón de encendido, necesito ayuda urgentemente",
    "¿Hacen descuento si compro 15 licencias?",
    "Estoy muy molesto: llegó tarde y la caja rota, pésimo servicio"
]

print("\nClasificación de mensajes recientes:\n")
for t, e, r in enrutar_mensajes(nuevos):
    print(f"→ '{t}' | tipo='{e}' | destino='{r}'")

joblib.dump(pipe, "pipeline_triage.joblib")
print("\nModelo almacenado en: pipeline_triage.joblib")

loaded = joblib.load("pipeline_triage.joblib")
print("Prueba tras recargar modelo:", loaded.predict(["No puedo entrar a mi cuenta, sale error 500"])[0])
