import pandas as pd
from datetime import datetime
from joblib import load
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Conexión al backend GraphQL
transport = RequestsHTTPTransport(
    url="https://ms-contabilidad.onrender.com/graphql",
    verify=False,
    retries=3,
)

client = Client(transport=transport, fetch_schema_from_transport=True)

# 2. Consulta GraphQL para obtener datos
query = gql("""
query {
  customers {
    id
    invoices {
      totalAmount
      dueDate
      status
    }
    payments {
      amount
      paymentDate
    }
  }
}
""")

data = client.execute(query)
clientes = data['customers']

# 3. Procesar datos igual que en entrenamiento
rows = []
for cliente in clientes:
    if not cliente['invoices']:
        continue

    total_facturas = sum(float(f['totalAmount']) for f in cliente['invoices'])
    total_pagado = sum(float(p['amount']) for p in cliente['payments'])

    factura = cliente['invoices'][0]
    due_date_str = factura.get('dueDate')
    if due_date_str:
        fecha_vencimiento = datetime.strptime(due_date_str, "%Y-%m-%d")
        dias_hasta_vencimiento = (fecha_vencimiento - datetime.now()).days
    else:
        dias_hasta_vencimiento = 0

    atrasado = 1 if factura['status'] == "OVERDUE" else 0

    rows.append({
        "total_facturas": total_facturas,
        "total_pagado": total_pagado,
        "dias_hasta_vencimiento": dias_hasta_vencimiento,
        "atrasado": atrasado
    })

df = pd.DataFrame(rows)

# 4. Preparar variables para test
X = df[["total_facturas", "total_pagado", "dias_hasta_vencimiento"]]
y = df["atrasado"]

# Si quieres separar un test, aquí puedes hacerlo; si no, puedes evaluar directo con todo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 5. Cargar modelo entrenado
modelo = load('modelo_overdue.joblib')

# 6. Predecir y evaluar
y_pred = modelo.predict(X_test)

print("=== Reporte de clasificación ===")
print(classification_report(y_test, y_pred))

print("=== Matriz de confusión ===")
print(confusion_matrix(y_test, y_pred))
