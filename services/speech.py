import json

import openai
import requests
from typing import List


class SpeechResponseEngine:

    def __init__(self):
        self.sourceURL = "<your web demo key>"
        self.payload_template = {"queryInput": {"text": {"text": "hello", "languageCode": "en"}}}
        openai.api_key = '<your open ai key>'

    def _make_payload(self, query: str) -> dict:
        payload = self.payload_template.copy()
        payload["queryInput"]["text"]["text"] = query
        return payload

    def get_response(self, query: str):
        payload = self._make_payload(query)
        res = requests.post(self.sourceURL, json=payload)
        responseText = ""
        b_audio = ""

        if res.status_code == 200:
            data = json.loads(res.text[5:])
            is_fallback = False

            if "intent" in data["queryResult"].keys() and "isFallback" in data["queryResult"]["intent"].keys():
                is_fallback = data["queryResult"]["intent"]["isFallback"]

            if is_fallback:
                responseText = self.get_response_chat_gpt(query)
                #responseText = "Chat GPT response"
                b_audio = self.get_binary_audio(responseText)

            else:
                responseText = data["queryResult"]["fulfillmentMessages"][0]["text"]["text"][0]
                b_audio = data["outputAudio"]

            return is_fallback, responseText, b_audio
        return None

    def get_response_chat_gpt(self, query: str) -> str:
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": "You are a kind, sweet and helpful assistant"},
                {"role": "system", "content": "you talks in about 30 word sentence"},
                {"role": "system", "content": "you talk like Jarvis from Ironman"},
                {"role": "user", "content": query}])

        responseText = chat.choices[0].message.content
        return responseText

    def get_binary_audio(self, text: str):
        payload = self._make_payload(f"speak out {text}")
        res = requests.post(self.sourceURL, json=payload)
        if res.status_code == 200:
            data = json.loads(res.text[5:])
            b_audio = data["outputAudio"]
            return b_audio

        return ""

    def make_sentence(self, words: List[str]):
        joined = ', '.join(words)
        res = self.get_response_chat_gpt(f"make a simple sentence from tokens {joined}")
        b_audio = self.get_binary_audio(res)

        return 200, res, b_audio
