from langchain_community.chat_models.tongyi import ChatTongyi


def build_chat_model(model_name: str = "qwen3-max"):
    return ChatTongyi(model=model_name)


chat_model = build_chat_model()

