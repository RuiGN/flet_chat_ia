import os

from typing import List

from decouple import config
from langchain_groq import ChatGroq

from components import Message


os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY')


class AIBot:

    def __init__(self):
        self.__llm = ChatGroq(model='llama-3.2-90b-vision-preview')

    def __build_history_messages(self, history_messages: List[Message]):
        messages = []
        for message in history_messages:
            messages.append(
                (
                    message.user_type,
                    message.text,
                ),
            )
        return messages if messages else None

    def __build_messages(self, history_messages, user_message):
        messages = [
            'system',
            'Você é um assistente responsável por tirar dúvidas sobre programação Python.'
            'Responda em Markdown'
        ]
        session_messages = self.__build_history_messages(history_messages=history_messages)
        if session_messages:
            messages.extend(session_messages)
        messages.append(
            (
                'user',
                user_message,
            )
        )
        return messages

    def invoke(self, history_messages, user_message):
        messages = self.__build_messages(
            history_messages=history_messages,
            user_message=user_message,
        )
        return self.__llm.invoke(messages).content
