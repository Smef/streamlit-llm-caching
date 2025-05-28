class Chat:

    chat_history = []

    def add_chat_history(self, role, message):
        self.chat_history.append({
            'message': message,
            'role': role
        })

    def get_chat_history(self):
        return self.chat_history
