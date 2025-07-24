from flask import Flask, render_template, jsonify
from graphql_client import GraphQLClient
import numpy as np
from datetime import datetime
from joblib import load
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
client = GraphQLClient()

# Queries base
QUERY_TOTAL_CLIENTES = """query { activeCustomers { id } }"""
QUERY_FACTURAS_ESTADOS = """query { customerInvoices { status } }"""
QUERY_FACTURACION_POR_PERIODO = """query {
  customerInvoices {
    totalAmount
    accountingPeriod { id }
  }
}"""
QUERY_PAGOS_METODO = """query { customerPayments { amount paymentMethod } }"""
QUERY_CLIENTES_VENCIDOS = """query {
  overdueInvoices {
    customer { id name }
  }
}"""
QUERY_CUENTAS = """query { accountingAccounts { id name } }"""
QUERY_SALDO_CUENTA = """query($accountId: ID!) {
  accountBalance(accountId: $accountId) { balance }
}"""

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# Endpoints desacoplados para KPIs
@app.route("/api/total_clientes")
def total_clientes():
    data = client.execute(QUERY_TOTAL_CLIENTES)
    total = len(data["activeCustomers"])
    return jsonify(total)

@app.route("/api/facturas_estado")
def facturas_estado():
    data = client.execute(QUERY_FACTURAS_ESTADOS)
    conteo = {"PENDING": 0, "PAID": 0, "CANCELLED": 0, "OVERDUE": 0}
    for inv in data["customerInvoices"]:
        conteo[inv["status"]] += 1
    return jsonify(conteo)

@app.route("/api/facturacion_periodos")
def facturacion_periodos():
    data = client.execute(QUERY_FACTURACION_POR_PERIODO)
    acumulado = {}
    for inv in data["customerInvoices"]:
        pid = inv["accountingPeriod"]["id"]
        acumulado[pid] = acumulado.get(pid, 0) + float(inv["totalAmount"])
    return jsonify(acumulado)

@app.route("/api/pagos_metodo")
def pagos_metodo():
    data = client.execute(QUERY_PAGOS_METODO)
    pagos = {}
    for p in data["customerPayments"]:
        metodo = p["paymentMethod"]
        pagos[metodo] = pagos.get(metodo, 0) + float(p["amount"])
    return jsonify(pagos)

@app.route("/api/clientes_vencidos")
def clientes_vencidos():
    data = client.execute(QUERY_CLIENTES_VENCIDOS)
    nombres = [inv["customer"]["name"] for inv in data["overdueInvoices"]]
    return jsonify(nombres)

@app.route("/api/top_saldos")
def top_saldos():
    data_cuentas = client.execute(QUERY_CUENTAS)["accountingAccounts"]
    cuentas_filtradas = data_cuentas[:5]

    def obtener_saldo(cuenta):
        try:
            saldo_data = client.execute(QUERY_SALDO_CUENTA, variables={"accountId": cuenta["id"]})
            return cuenta["name"], float(saldo_data["accountBalance"]["balance"])
        except:
            return cuenta["name"], 0.0

    with ThreadPoolExecutor(max_workers=5) as executor:
        resultados = executor.map(obtener_saldo, cuentas_filtradas)

    cuentas_saldo = dict(resultados)
    return jsonify(cuentas_saldo)

@app.route("/api/prediccion_riesgo")
def riesgo_clientes():
    query = """{
      customers {
        id
        name
        invoices { totalAmount dueDate status }
        payments { amount paymentDate }
      }
    }"""
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

    rows.sort(key=lambda r: r["riesgo"], reverse=True)
    top_10 = rows[:10]
    return jsonify(top_10)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)