import requests

class Fish:
    def __init__(self, url, sysmessage):
        self.url = url
        self.sysmessage = sysmessage
    
    def generate(self, payload):
        data = {
            "model": "Fish:1.5",
            "messages": [
                {"role": "system","content": f"{self.sysmessage}"},
                {"role": "user","content": f"{payload}"},
            ],
            "stream": False
        }
        response = requests.post(self.url, json=data)
        dict = response.json()
        reply = dict["message"]["content"]
        return reply