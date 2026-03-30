import ollama


USER_ROLE:str = 'user'
ASSISTANT_ROLE:str = 'assistant'

class ChatBot:

    def __init__(self):
        self.messages: list[dict] = []

    def send(self, new_mesg) -> str:
        self._chat_as(USER_ROLE, new_mesg)
        return self._process()

    def _process(self) -> str:
        response = ollama.chat(
            model='qwen2.5',
            messages=self.messages
        )
        answer:str = response['message']['content']
        self._chat_as(ASSISTANT_ROLE, answer)
        return answer

    def _chat_as(self, role:str, content: str) -> None:
        new_msg = {
            "role": role,
            "content": content
        }
        self.messages.append(new_msg)


def main():
    chatbot = ChatBot()
    is_working:bool = True
    print('este es un chatbot empieza a interactuar cuando termines escribe "exit"')
    while is_working:
        new_msg = input('> ')
        is_working = new_msg != 'exit'
        if is_working:
            answer:str = chatbot.send(new_mesg=new_msg)
            print(f'\n*.- {answer}')
