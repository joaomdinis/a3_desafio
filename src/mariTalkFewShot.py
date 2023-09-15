from src.mariTalkAPI import MariTalkAPI

class MariTalkFewShot():
    def __init__(self, api_key, messages, max_tokens=50, do_sample=False, temperature=0.7, top_p=0.95):
        self.api_key = api_key
        self.messages = messages
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.model = MariTalkAPI(self.api_key)

    def predict(self):
        body = {
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "do_sample": self.do_sample,
            "model": "maritalk",
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        return self.model.inference(body)
