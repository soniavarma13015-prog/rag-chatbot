from backend.config import config

_store: dict[str, list[dict]] = {}


class ChatHistoryManager:
    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        if session_id not in _store:
            _store[session_id] = []
        _store[session_id].append({"role": "user", "content": user_msg})
        _store[session_id].append({"role": "assistant", "content": assistant_msg})
        max_messages = config.max_history_turns * 2
        if len(_store[session_id]) > max_messages:
            _store[session_id] = _store[session_id][-max_messages:]

    def get_history(self, session_id: str) -> list[dict]:
        return list(_store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        _store.pop(session_id, None)


chat_history_manager = ChatHistoryManager()