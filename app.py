from flask import Flask, render_template, jsonify
from graphql_client import GraphQLClient
import numpy as np
from datetime import datetime
from joblib import load
from gql import gql
from gql.transport.requests import RequestsHTTPTransport
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
client = GraphQLClient()

# Queries base
QUERY_TOTAL_CLIENTES = """
query { activeCustomers { id } }
"""

QUERY_FACTURAS_ESTADOS = """
query {
  customerInvoices {
    status
  }
}
"""

QUERY_FACTURACION_POR_PERIODO = """
query {
  accountingPeriods {
    id
    name
  }
  customerInvoices {
    totalAmount
    accountingPeriod {
      id
    }
  }
}
"""

QUERY_PAGOS_METODO = """
query {
  customerPayments {
    amount
    paymentMethod
  }
}
"""

QUERY_CLIENTES_VENCIDOS = """
query {
  overdueInvoices {
    customer {
      id
      name
    }
  }
}
"""

QUERY_CUENTAS = """query { accountingAccounts { id name } }"""
QUERY_SALDO_CUENTA = """query($accountId: ID!) {
  accountBalance(accountId: $accountId) {
    balance
  }
}"""

@app.route("/api/kpis")
def get_kpis():
    # Total clientes activos
    data_clients = client.execute(QUERY_TOTAL_CLIENTES)
    total_clients = len(data_clients["activeCustomers"])

    # Facturas por estado
    data_invoices = client.execute(QUERY_FACTURAS_ESTADOS)
    invoices = data_invoices["customerInvoices"]

    estados_count = {"PENDING": 0, "PAID": 0, "CANCELLED": 0, "OVERDUE": 0}
    for inv in invoices:
        estados_count[inv["status"]] += 1

    # Facturación por período
    data_periodos = client.execute(QUERY_FACTURACION_POR_PERIODO)
    periodos = {p["id"]: p["name"] for p in data_periodos["accountingPeriods"]}
    invoices_por_periodo = {}
    for inv in data_periodos["customerInvoices"]:
        pid = inv["accountingPeriod"]["id"]
        invoices_por_periodo[pid] = invoices_por_periodo.get(pid, 0) + float(inv["totalAmount"])

    # Pagos por método
    data_pagos = client.execute(QUERY_PAGOS_METODO)
    pagos = data_pagos["customerPayments"]
    pagos_por_metodo = {}
    for p in pagos:
        metodo = p["paymentMethod"]
        pagos_por_metodo[metodo] = pagos_por_metodo.get(metodo, 0) + float(p["amount"])

    # Clientes con facturas vencidas (únicos)
    data_vencidos = client.execute(QUERY_CLIENTES_VENCIDOS)
    clientes_vencidos = {inv["customer"]["id"]: inv["customer"]["name"] for inv in data_vencidos["overdueInvoices"]}
    total_clientes_vencidos = len(clientes_vencidos)

    
    data_cuentas = client.execute(QUERY_CUENTAS)["accountingAccounts"]
    def obtener_saldo(cuenta):
            try:
                saldo_data = client.execute(QUERY_SALDO_CUENTA, variables={"accountId": cuenta["id"]})
                saldo = float(saldo_data["accountBalance"]["balance"])
                return cuenta["name"], saldo
            except Exception:
                return cuenta["name"], 0.0

    cuentas_saldo = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(obtener_saldo, c) for c in data_cuentas]
        for future in as_completed(futures):
            nombre, saldo = future.result()
            cuentas_saldo[nombre] = saldo

    # Top 5
    cuentas_top = dict(sorted(cuentas_saldo.items(), key=lambda item: abs(item[1]), reverse=True)[:5])

    return jsonify({
        "total_clients": total_clients,
        "invoices_status_count": estados_count,
        "invoices_by_period": invoices_por_periodo,
        "payments_by_method": pagos_por_metodo,
        "overdue_customers_count": total_clientes_vencidos,
        "overdue_customers_list": list(clientes_vencidos.values()),
        "top_account_balances": cuentas_top
    })

# En dashboard, pasarás estos datos para visualizar con Plotly (ejemplo abreviado)
@app.route("/")
def dashboard():
    kpis = get_kpis().json

    # Prepara datos para gráficos
    invoices_status = kpis["invoices_status_count"]
    invoices_by_period = kpis["invoices_by_period"]
    payments_by_method = kpis["payments_by_method"]

    # Mapear IDs de período a nombres para el gráfico de barras
    # (Si prefieres, pasa también nombres del periodo en la API)
    # Aquí para ejemplo simplificado pasamos tal cual
    return render_template("dashboard.html",
                           invoices_status=invoices_status,
                           invoices_by_period=invoices_by_period,
                           payments_by_method=payments_by_method,
                           overdue_customers=kpis["overdue_customers_list"],
                           total_clients=kpis["total_clients"],
                           top_account_balances=kpis["top_account_balances"])



@app.route("/api/prediccion_riesgo")
def riesgo_clientes():
    query = """
    {
    customers {
        id
        name
        invoices { totalAmount dueDate status }
        payments { amount paymentDate }
    }
    }
    """



    data = client.execute(query)
    clientes = data["customers"]

    modelo = load("modelo_overdue.joblib")
    rows = []

    for cliente in clientes:
        if not cliente["invoices"]:
            continue

        total_facturas = sum(float(f["totalAmount"]) for f in cliente["invoices"])
        total_pagado = sum(float(p["amount"]) for p in cliente["payments"])
        factura = cliente["invoices"][0]
        due_date = factura.get("dueDate")

        dias_hasta_vencimiento = 0
        if due_date:
            fecha_vencimiento = datetime.strptime(due_date, "%Y-%m-%d")
            dias_hasta_vencimiento = (fecha_vencimiento - datetime.now()).days

        X = np.array([[total_facturas, total_pagado, dias_hasta_vencimiento]])
        prob = modelo.predict_proba(X)[0][1]

        rows.append({
            "id": cliente["id"],
            "name": cliente["name"],
            "riesgo": round(prob, 3)
        })

    # Ordenar y devolver top 10
    rows.sort(key=lambda r: r["riesgo"], reverse=True)
    top_10 = rows[:10]

    return jsonify(top_10)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

