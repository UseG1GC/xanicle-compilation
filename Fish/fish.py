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

class Judge:
    def __init__(self, url, sysmessage="You are an AI fish. You will be given a scenario and what a player tries to do, and you will come up with a result.", limit = 20):
        self.url = url
        self.sysmessage = sysmessage
        self.messages = [{"role": "system","content": f"{self.sysmessage}"}]
        self.limit = limit

    def generate(self, payload):
        self.messages.append({"role": "user","content": f"{payload}"})
        data = {
            "model": "Fish:1.5",
            "messages": self.messages,
            "stream": False
        }
        response = requests.post(self.url, json=data)
        dict = response.json()
        reply = dict["message"]["content"]
        self.messages.append(dict["message"])
        if len(self.messages) > self.limit:
            del self.messages[1:2]
        return reply

    def scenario(self, message):
        self.messages[0]["content"] = f"{self.sysmessage}\n\nScenario: {message}"