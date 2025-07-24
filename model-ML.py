import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

transport = RequestsHTTPTransport(
    url="https://ms-contabilidad.onrender.com/graphql",
    verify=False,
    retries=3,
)

client = Client(transport=transport, fetch_schema_from_transport=True)

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

rows = []
for cliente in clientes:
    if not cliente['invoices']:
        continue  # ignorar clientes sin facturas
    
    total_facturas = sum(float(f['totalAmount']) for f in cliente['invoices'])
    total_pagado = sum(float(p['amount']) for p in cliente['payments'])

    factura = cliente['invoices'][0]
    due_date_str = factura.get('dueDate')
    if due_date_str:
        fecha_vencimiento = datetime.strptime(due_date_str, "%Y-%m-%d")
        dias_hasta_vencimiento = (fecha_vencimiento - datetime.now()).days
    else:
        dias_hasta_vencimiento = 0

    pagado = 1 if factura['status'] == "OVERDUE" else 0

    rows.append({
        "total_facturas": total_facturas,
        "total_pagado": total_pagado,
        "dias_hasta_vencimiento": dias_hasta_vencimiento,
        "atrasado": pagado
    })

df = pd.DataFrame(rows)

X = df[["total_facturas", "total_pagado", "dias_hasta_vencimiento"]]
y = df["atrasado"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

dump(modelo, 'modelo_overdue.joblib')

print("✅ Modelo entrenado y guardado como 'modelo_overdue.joblib'")
