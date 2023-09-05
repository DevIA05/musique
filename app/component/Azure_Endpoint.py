import requests
import urllib.request
import json
import os
import ssl

# Effectue une prédiction de la fertilité à partir de valeur et traitre la 
# réponse obtenu 
class AutoML:
    def __init__(self):
        self.allowSelfSignedHttps(True) 
        self._url = os.environ['API_URL']
        self._api_key = os.environ['API_KEY']
        self._azureml_model_deployment = os.environ['API_MODEL']
        
    def allowSelfSignedHttps(self, allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context
    
    def getPred(self, data):  
        """
        Effectue une prédiction en utilisant les données spécifiées en envoyant 
        une requête à l'API de prédiction.
        Args:
            data (dict): Les données à prédire.
        Returns:
            str: Catégorie prédite.
        """
        
        #self.allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
        # Request data goes here
        # The example below assumes JSON formatting which may be updated
        # depending on the format your endpoint expects.
        # More information can be found here:
        # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
        data_processed = json.loads(data.to_json(orient='records'))
        #print(data_processed)
        content =  {
          "Inputs": {
            "data": data_processed 
          },
          "GlobalParameters": {
            "method": "predict"
          }
        }

        body = str.encode(json.dumps(content))
        headers = {'Content-Type':'application/json', 
                   'Authorization': ('Bearer ' + self._api_key), 
                   'azureml-model-deployment': self._azureml_model_deployment }
        req = urllib.request.Request(self._url, body, headers)
        result = None
        try:
            response = urllib.request.urlopen(req)
            response = response.read()
            # Charger le JSON dans un dictionnaire Python
            result = json.loads(response)
            # Accéder à la valeur de "Results"
            result = result["Results"][0]
        except urllib.error.HTTPError as error:
            #print("The request failed with status code: " + str(error.code))
            result = str(error.code)
            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            #print(error.info())
            #print(error.read().decode("utf8", 'ignore'))
            
        return result