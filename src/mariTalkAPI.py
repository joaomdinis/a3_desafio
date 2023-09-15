import requests
from requests.exceptions import HTTPError

class MariTalkAPI(object):
    def __init__(self, APIKey):
        self.endpoint = "https://chat.maritaca.ai"
        self.APIKey = APIKey
        self.headers = {"authorization": f"Key {self.APIKey}"}
    
    def _sendPost(self, endpoint, body):
        try:
            post = requests.post(url=endpoint, json=body, headers=self.headers)
            self.status_code = post.status_code
            return post.json()
            
        except HTTPError as e:
            return e
        
    def inference(self, body):
        endpoint = self.endpoint + '/api/chat/inference'
        return self._sendPost(endpoint, body)   