import requests

GRAPHQL_URL = "https://ms-contabilidad.onrender.com/graphql"

class GraphQLClient:
    def __init__(self, url=GRAPHQL_URL):
        self.url = url

    def execute(self, query: str, variables: dict = None):
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")

        return result["data"]
